
require 'nn'
require 'rnn'

local storn = {}

function storn.build_encoder(input_size, hidden_size, latent_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false
	local dropout = dropout or false

	local encoder = nn.Sequential()

	local rm = nn.Sequential()
	
	for layer=1, n_layers do 

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		local lstm
		if layer == 1 then
		 rm:add(nn.FastLSTM(input_size, hidden_size))
		else
		 rm:add(nn.FastLSTM(hidden_size, hidden_size))
		end

		if stabilise then
			rm:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	encoder:add(nn.Sequencer(rm))

	--adding mean and logvar output
	local mean = nn.Linear(hidden_size, latent_size)
	local logvar = nn.Linear(hidden_size, latent_size)
	local means_logvars = nn.ConcatTable()
		:add(nn.Sequencer(mean))
		:add(nn.Sequencer(logvar))

	encoder:add(means_logvars)

	return encoder

end 



function storn.build_decoder(latent_size, hidden_size, output_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false
	local dropout = dropout or false

	local decoder = nn.Sequential()

	local rm = nn.Sequential()

	for layer=1, n_layers do

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
		 rm:add(nn.FastLSTM(latent_size, hidden_size))
		else
		 rm:add(nn.FastLSTM(hidden_size, hidden_size))
		end
		
		--adding stabiling layer
		if stabilise then
			rm:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			rm:add(nn.Dropout(0.5))
		end
	end

	rm:add(nn.Linear(hidden_size, output_size))
	rm:add(nn.Sigmoid(true))

	decoder:add(nn.Sequencer(rm))


	return decoder

end

return storn