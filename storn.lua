
require 'nn'
require 'rnn'

local storn = {}

function storn.build_encoder(input_size, hidden_size, latent_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false

	local encoder = nn.Sequential()

	local lstms = nn.Sequential()
	
	for layer=1, n_layers do 

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		local lstm
		if layer == 1 then
		 lstm = nn.FastLSTM(input_size, hidden_size)
		else
		 lstm = nn.FastLSTM(hidden_size, hidden_size)
		end
		lstms:add(lstm)

		if stabilise then
			lstms:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			lstms:add(nn.Dropout(0.5))
		end

	end

	encoder:add(nn.Sequencer(lstms))

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

	local decoder = nn.Sequential()

	local lstms = nn.Sequential()

	for layer=1, n_layers do

		--adding lstm layer
		local lstm
		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
		 lstm = nn.FastLSTM(latent_size, hidden_size)
		else
		 lstm = nn.FastLSTM(hidden_size, hidden_size)
		end
		lstms:add(lstm)
		
		--adding stabiling layer
		if stabilise then
			lstms:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			lstms:add(nn.Dropout(0.5))
		end
	end

	lstms:add(nn.Linear(hidden_size, output_size))
	lstms:add(nn.Sigmoid(true))

	decoder:add(nn.Sequencer(lstms))


	return decoder

end

return storn