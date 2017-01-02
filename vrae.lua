
require 'nn'
require 'rnn'

local vrae = {}

function vrae.build_encoder(input_size, hidden_size, latent_size, n_layers, dropout, batch_norm, stabilise)
	
	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false

	local encoder = nn.Sequential()

	local rm = nn.Sequential()
	
	for layer=1, n_layers do 

		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
			rm:add(nn.FastLSTM(input_size, hidden_size))
		elseif layer == n_layers then
			encoder.last_lstm = nn.FastLSTM(hidden_size, hidden_size)
			rm:add(encoder.last_lstm)
		else
		 	rm:add(nn.FastLSTM(hidden_size, hidden_size))
		end

		if stabilise then
			rm:add(nn.NormStabilizer())
		end

		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	encoder:add(nn.Sequencer(rm))

	--adding mean and logvar
	encoder:add(nn.Select(1, -1))
	encoder:add(nn.ConcatTable()
			:add(nn.Linear(hidden_size, latent_size))
			:add(nn.Linear(hidden_size, latent_size)))

	return encoder
end 




function vrae.build_decoder(input_size, hidden_size, output_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false
	
	local decoder = nn.Sequential()
	
	local rm = nn.Sequential()
	
	for layer=1, n_layers do 

		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
			decoder.first_lstm = nn.FastLSTM(input_size, hidden_size)
			rm:add(decoder.first_lstm)
		else
			rm:add(nn.FastLSTM(hidden_size, hidden_size))
		end

		if stabilise then
			rm:add(nn.NormStabilizer(rm))
		end

		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	rm:add(nn.Linear(hidden_size, output_size))
	rm:add(nn.Sigmoid(true))

	decoder:add(nn.Sequencer(rm))

	function decoder.set_init_cell(cell_state)
		decoder.first_lstm.userPrevCell =
			nn.rnn.recursiveCopy(decoder.first_lstm.userPrevCell, cell_state)
	end

	function decoder.copy_init_grad_cell(init_grad_cell)
		init_grad_cell = 
			nn.rnn.recursiveCopy(init_grad_cell, decoder.first_lstm.userGradPrevCell)
	end


	function decoder.sample(init_cell_state, rho)

		decoder:evaluate()
		decoder.set_init_cell(init_cell_state)

		local outputs = {}
		local inputSize = decoder.first_lstm.inputSize
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

--[[
		decoder:evaluate()
		decoder.set_init_cell(init_cell_state)

		local outputs = {}
		local inputSize = decoder.first_lstm.inputSize
		local outputs = {[0] = }
		local inputs = torch.Tensor():resize(1, 1, inputSize):zero()
		for t=1, rho do
			local output = decoder:forward(inputs)

			inputs

			inputs = torch.cat(outputs, 1)
			outputs[t] = output
			--outputs:resize(t+1, 1, inputSize):select(1,t):copy(output[t])
		end
		
		decoder:training()
		
		return torch.cat(outputs, 1)
]]
