
require 'nn'
require 'rnn'

local lstm = {}

function lstm.build(input_size, hidden_size, output_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false

	local modules = nn.Sequential()

	for layer=1, n_layers do 

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
		 modules:add(nn.FastLSTM(input_size, hidden_size))
		else
		 modules:add(nn.FastLSTM(hidden_size, hidden_size))
		end

		--adding stabilising regularization
		if stabilise then
			modules:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			modules:add(nn.Dropout(0.5))
		end

	end

	 modules:add(nn.Linear(hidden_size, output_size))
	 modules:add(nn.Sigmoid(true))

	return nn.Sequencer(modules)

end 

return lstm