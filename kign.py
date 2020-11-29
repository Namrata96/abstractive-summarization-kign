import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module

class EncoderRNN(nn.Module):
	def __init__(self, input_size, hidden_size, wordEmbed):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding_layer = wordEmbed
		self.biGRU_layer = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

	def forward(self, input, hidden):
		batch_size, max_len = input.size(0), input.size(1)
		embedding_layer_output = self.embedding_layer(input).view(1,1,-1)
		mask = input.eq(0).detach()
		inv_mask = mask.eq(0).unsqueeze(2).expand(batch_size, max_len, self.hidden_size * 2).float().detach()
		biGRU_hidden, biGRU_state = self.biGRU_layer(embedding_layer_output)
		biGRU_hidden = biGRU_hidden * inv_mask # not including mask in the h_t states for each state, size (batch_size, seq_len, num_directions*hidden_size)
		return biGRU_hidden, biGRU_state, mask

# Used for the key representation.
class KeyEncoder(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size, wordEmbed):
		super(EncoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.embedding_layer = wordEmbed
		self.biGRU_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)

	def forward(self, input):
		batch_size, num_keywords = input.size()
		embedding_layer_output = self.embedding_layer(input).view(1,1,-1)
		# no mask as the input will just be keywords one after another
		biGRU_hidden, biGRU_state = self.biGRU_layer(embedding_layer_output)
		key_representation = torch.cat((biGRU_hidden[:,num_keywords,self.hidden_size], biGRU_hidden[:,0,self.hidden_size:]),1)
		return torch.cat(key_representation*batch_size).view(batch_size, 2*self.hidden_size)
        
class PointerAttentionDecoder(nn.Module):
	def __init__(self, input_size, hidden_size, vocab_size, wordEmbed):
		super(PointerAttentionDecoder, self).__init__()
		self.hidden_size = hidden_size
		self.embedding_layer = wordEmbed
		self.GRU_layer = nn.GRUCell(input_size=input_size, hidden_size=hidden_size, batch_first=True)
		self.Wh_layer = nn.Linear(in_features=2*hidden_size, out_features=2*hidden_size) #biGRU
		self.Ws_layer = nn.Linear(in_features=hidden_size, out_features=2*hidden_size)
		self.Wk_layer = nn.Linear(in_features=2*hidden_size, out_features=2*hidden_size)
		self.Wv_layer = nn.Linear(in_features=2*hidden_size,out_features=1)
		self.softmax_layer = nn.Softmax(2*hidden_size)
		self.context_vector_layer = nn.Linear(in_features=2*hidden_size)

		self.wf_layer = nn.Linear(in_features=5*hidden_size, out_features=vocab_size)
		self.wc_layer = nn.Linear(in_features=2*hidden_size, out_features=1) #biGRU
		self.ws_layer = nn.Linear(in_features=hidden_size, out_features=1)
		self.wk_layer = nn.Linear(in_features=2*hidden_size, out_features=1)
		self.loss = []
        
         ## variables for prediction guide network
         cos = nn.CosineSimilarity(dim=1, eps=1e-8)
         # Using this for storing the last decoder state after training kign to intialize the decoder again in value network.
		self.dec_state_after_kign = torch.Variable(torch.zeros(1, batch_size, 2*self.hidden_size))
         self.linear_layer = nn.Linear(in_features=2*hidden_size, out_features=1)
         self.sigmoid_layer = nn.Sigmoid()
        
	def setValues(self, start_id, stop_id, unk_id):
		# start/stop tokens
		self.start_id = start_id
		self.stop_id = stop_id
		self.unk_id = unk_id
		# max_article_oov -> max number of OOV in articles i.e. enc inputs. Will be set for each batch individually
		self.max_article_oov = None
		
	def forward(self, enc_hidden_states, enc_state, enc_mask, dec_input, article_inds, targets=None, key_representation=None, train_value_network=False):
		if train_value_network is True:
            # Here dec input will be the training data only
            return value_network(enc_hidden_states, enc_state, enc_mask, dec_input, article_inds, targets, key_representation)
         embedding_layer_output = self.embedding_layer(dec_input).view(1,1,-1)
		batch_size, max_enc_len, enc_size = enc_hidden_states.size() # enc_size = 2 * hidden_size
		max_dec_len = dec_input.size(1)  
        dec_lens = (dec_input > 0).float().sum(1)               
		context_vector = torch.Variable(torch.zeros(batch_size, 2*self.hidden_size))
		Wk_layer_output = self.Wk_layer(key_representation)
		encoder_features = self.Wh_layer(enc_hidden_states.view(batch_size*max_enc_len, enc_size)).view(batch_size, max_enc_len, None)
		dec_state = enc_state
		all_attn_dist = []
		all_psw = []

		for time_step in range(max_dec_len):
			dec_hidden_timestep, dec_state = self.GRU_layer(embedding_layer_output[time_step].add(context_vector), dec_state) # assuming both embedding dim and context vector to be of size 2*hidden_size
			decoder_features = self.Ws_layer(dec_hidden_timestep) # size batch, 2*hidden_size
			 
			attn_weights = self.Wv_layer(F.tanh(encoder_features + Wk_layer_output + decoder_features)).view(batch_size*max_enc_len, None) # size (batch_size*max_enc_len, None)
			attn_weights = attn_weights.view(batch_size, max_enc_len)
			attn_weights.masked_fill_(enc_mask, -float('inf')) # size batch_size, max_enc_len
			  
			attn_dist = self.softmax_layer(attn_weights) # size batch_size, max_enc_len
			all_attn_dist.append(attn_dist)
			context_vector = attn_dist.unsqueeze(1).bmm(enc_hidden_states).squeeze(1) # multiplying (b,1,max_len) with (b, max_len, 2*hidden_size) = (b, 1, 2*hidden_size) therefore we squeeze at pos 1 to get shape (b,2*hidden_size)
			p_sw = F.sigmoid(self.ws_layer(dec_hidden_timestep) + self.wc_layer(context_vector) + self.wk_layer(key_representation)) # shape (batch_size, 1)
			p_vocab = F.softmax(self.wf_layer(torch.cat((dec_hidden_timestep, torch.cat((context_vector, key_representation), 1),1)))) # input to wf_layer is of shape (batch_size, 5*hidden_size) as dec_hidden_timestep is of shape(b,hidden_size), rest are (b,2*hidden_size)
			# pvocab shape (batch_size, vocab_size)
			p_sw = p_sw.view(None,1)
			 # just to ensure the correct shape of p_sw
			all_psw.append(p_sw)
			weighted_pvocab = p_vocab * p_sw # multiplying the p_sw value elementwise with matrix contents of p_vocab. output shape (batch_size, vocab_size)
			weighted_attn = (1-p_sw) * attn_dist # same as above. output shape (batch_size, max_enc_len)

			if self.max_article_oov > 0:
			   extend_vocab = torch.Variable(torch.zeros(batch_size, self.max_article_oov).cuda())
			   combined_vocab = torch.cat((weighted_pvocab, extend_vocab), 1) # shape (batch_size, vocab_size + max_article_oov)
			   del extend_vocab
			else:
			   combined_vocab = weighted_pvocab
			   
			del weighted_pvocab
  
			# -1 for accounting 0-index
			article_inds_masked = article_inds.add(-1).masked_fill_(enc_mask, 0)
			combined_vocab = combined_vocab.scatter_add(1, article_inds_masked, weighted_attn)	 # adding weighted attention values to combined pvocab at masked article indexes in 1st dimension (as 0 dim is batch_size)					

			 
			target = targets[:, time_step].unsqueeze(1)
			# mask the output to account for PAD
			# subtract one from target for 0-indexarget_mask_0 = target.ne(0).detach()	
			target_mask_0 = target.ne(0).detach()	
			target_mask_p = target.eq(0).detach()
			target = target - 1
			output = combined_vocab.gather(1, target.masked_fill_(target_mask_p, 0))  			
			self.loss.append(output.log().mul(-1) * target_mask_0.float())
        self.dec_state_after_kign = dec_state
        total_loss = torch.cat(self.loss, 1).sum(1).div(dec_lens)
        return total_loss, all_attn_dist, all_psw

    def value_network(enc_hidden_states, enc_state, enc_mask, dec_input, article_inds, targets=None, key_representation):
        # Params for random stop. TODO: Verify if these stops are possible
        stop_t1 = 12
        stop_t2 = 35
        
        embedding_layer_output = self.embedding_layer(dec_input).view(1,1,-1)
	    batch_size, max_enc_len, enc_size = enc_hidden_states.size() # enc_size = 2 * hidden_size
	    max_dec_len = dec_input.size(1)  
        dec_state = self.dec_state_after_kign
        
        dec_all_hidden = list() # shape of each entry will be [batch, timestep, 2*hidden_size]
        
        dec_state_t1 = dec_state
        dec_state_t2 = dec_state
        
        for time_step in range(stop_t2):
            dec_hidden_timestep, dec_state = self.GRU_layer(embedding_layer_output[time_step], dec_state)
            if time_step == stop_t1:
                dec_state_t1 = dec_state
            dec_common_hidden.append(dec_hidden_timestep)
        dec_state_t2 = dec_state
        # dec_hidden_all_t1 and t2 will have the remaining decoder hidden states after timestep stop_t1 and stop_t2 respectively
        # dec_hidden_all_t1 will have shape (batch, seq_len-stop_t1, 2*hidden_size, M)
        
        full_summaries_y1, dec_hidden_t1 = beam_search(dec_state_t1) # this will have M full summaries(each of 2*hidden_size) for y1 using beam search
        full_summaries_y2, dec_hidden_t2 = beam_search(dec_state_t2)
        
        # Calculations for AvgCos
        dec_value_t1 = torch.cat((dec_hidden_t1, dec_common_hidden.unsqueeze(3).repeat(1,1,1,M)), 2) # result shape will be [batch, max_dec_len_y1, 2*hidden_size, M]
        dec_value_t2 = torch.cat((dec_hidden_t2, dec_common_hidden.unsqueeze(3).repeat(1,1,1,M)), 2) # result shape will be [batch, max_dec_len_y2, 2*hidden_size, M]

        dec_value_avg_t1 = dec_value_t1.view(batch_size, -1, M).mean(1) # result shape is [batch, M]
        dec_value_avg_t2 = dec_value_t2.view(batch_size, -1, M).mean(1)
        
        cos_output_y1 = cos(dec_hidden_avg_t1,key_representation.expand(batch_size, )) # result is of the shape [batch, M]
        cos_output_y2 = cos(dec_hidden_avg_t2,key_representation)
        
        avg_cos_y1 = cos_output_y1.mean(1)
        avg_cos_y2 = cos_output_y2.mean(1)
        
        # Value network
        dec_common_hidden_avg_t1 = dec_common_hidden[:stop_t1].sum(1).div(stop_t1)
        dec_common_hidden_avg_t2 = dec_common_hidden.sum(1).div(len(dec_common_hidden))
        
        linear_layer_output_y1 = self.linear_layer(torch.cat((torch.cat((enc_hidden_states_avg, dec_value_avg_t1), 1), key_representation), 1)
        value_y1 = self.sigmoid_layer(linear_layer_output_y1) # shape (batch, 1)
        
        linear_layer_output_y2 = self.linear_layer(torch.cat((torch.cat((enc_hidden_states_avg, dec_value_avg_t2), 1), key_representation), 1)
        value_y2 = self.sigmoid_layer(linear_layer_output_y2) # shape (batch, 1)
        
        
        if avg_cos_y1 > avg_cos_y2:
            value_loss = torch.exp(value_y1 - value_y2).sum(1)
        else:
            value_loss = torch.exp(value_y2 - value_y1).sum(1)

        
        return value_loss
        
        
class SummaryNet(Module):
	def __init__(self, input_size, hidden_size, kign_input_size, kign_hidden_size, vocab_size, wordEmbed, start_id, stop_id, unk_id):
		super(SummaryNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.kign_input_size = kign_input_size
		self.kign_hidden_size = kign_hidden_size
		self.encoder = EncoderRNN(self.input_size, self.hidden_size, wordEmbed)
		self.kign = KeyEncoder(self.kign_input_size, self.kign_hidden_size. wordEmbed)
		self.pointerDecoder = PointerAttentionDecoder(self.input_size, self.hidden_size, vocab_size, wordEmbed)
		self.pointerDecoder.setValues(start_id, stop_id, unk_id)
         self.value_network = ValueNetwork()
	def forward(self, _input, keywords, max_article_oov, decode_flag=False, train_value_network=False):
		# set num article OOVs in decoder
        self.pointerDecoder.max_article_oov = max_article_oov		
        if train_value_network is False:
        		# train code for KIGN
        		enc_input, article_inds, rev_enc_input, dec_input, dec_target = _input
        		enc_hidden_states, enc_state, enc_mask = self.encoder(enc_input)
        		key_representation = self.kign(keywords)
        		total_loss, attn_dists, all_psw = self.pointerDecoder(enc_hidden_states, enc_state, enc_mask, dec_input, article_inds, targets=dec_target, key_representation=key_representation, train_value_network=False)
        		return total_loss, attn_dists, all_psw
        else:
            # train code for Prediction Guide Network
            enc_input, article_inds, rev_enc_input, dec_input, dec_target = _input
            enc_hidden_states, enc_state, enc_mask = self.encoder(enc_input)
            key_representation = self.kign(keywords)

            total_loss = self.pointerDecoder(enc_hidden_states, enc_state, enc_mask, dec_input, article_inds, targets=dec_target, key_representation=key_representation, train_value_network=True)
