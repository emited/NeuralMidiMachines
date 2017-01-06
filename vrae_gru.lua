
require 'nn'
require 'rnn'

local vrae = {}

function vrae.build_encoder(input_size, hidden_size, latent_size, n_layers, dropout)
	
	local n_layers = n_layers or 1
	local stabilise = stabilise or false
	local dropout = dropout or false

	local encoder = nn.Sequential()

	local rm = nn.Sequential()
	
	for layer=1, n_layers do 

		if layer == 1 then
			rm:add(nn.GRU(input_size, hidden_size))
		else
		 	rm:add(nn.GRU(hidden_size, hidden_size))
		end

		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	encoder:add(nn.Sequencer(rm))

	encoder:add(nn.Select(1, -1))
	encoder:add(nn.ConcatTable()
			:add(nn.Linear(hidden_size, latent_size))
			:add(nn.Linear(hidden_size, latent_size)))

	return encoder
end 




function vrae.build_decoder(input_size, hidden_size, output_size, n_layers, dropout)

	local n_layers = n_layers or 1
	local stabilise = stabilise or false
	local dropout = dropout or false

	local decoder = nn.Sequential()
	
	local rm = nn.Sequential()
	
	for layer=1, n_layers do 

		nn.GRU.bn = batch_norm
		if layer == 1 then
			decoder.first_gru = nn.GRU(input_size, hidden_size)
			rm:add(decoder.first_gru)
		else
			rm:add(nn.GRU(hidden_size, hidden_size))
		end

		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	rm:add(nn.Linear(hidden_size, output_size))
	rm:add(nn.Sigmoid(true))

	decoder:add(nn.Sequencer(rm))

	function decoder.set_init_h(init_h)
		decoder.first_gru.userPrevOutput =
			nn.rnn.recursiveCopy(decoder.first_gru.userPrevOutput, init_h)
	end

	function decoder.copy_init_grad_h(init_grad_h)
		init_grad_h = 
			nn.rnn.recursiveCopy(init_grad_h, decoder.first_gru.userGradPrevOutput)
		return init_grad_h
	end


	function decoder.sample(init_h, rho)

		decoder:evaluate()
		decoder.set_init_h(init_h)

		local outputs = {}
		local inputSize = decoder.first_gru.inputSize
		local outputs = {torch.Tensor():resize(1,inputSize):zero()}

		for t=1, rho do
			output = decoder:forward(outputs)[#outputs]
			table.insert(outputs, output)
		end

		decoder:training()
		return torch.cat(outputs, 1)
	end
	

	return decoder
end



return vrae