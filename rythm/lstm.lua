
require 'nn'
require 'rnn'

local lstm = {}

function lstm.build(input_size, hidden_size, note_size, offset_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false
	local dropout = dropout or false

	local rm = nn.Sequential()

	for layer=1, n_layers do 

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		if layer == 1 then
		 rm:add(nn.FastLSTM(input_size, hidden_size))
		else
		 rm:add(nn.FastLSTM(hidden_size, hidden_size))
		end

		--adding stabilising regularization
		if stabilise then
			rm:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	local notes = nn.Sequential()
		:add(nn.Linear(hidden_size, note_size))
		:add(nn.SoftMax())

	local offsets = nn.Sequential()
		:add(nn.Linear(hidden_size, offset_size))
		:add(nn.SoftMax())

	local notes_offsets = nn.ConcatTable()
		:add(nn.Sequencer(notes))
		:add(nn.Sequencer(offsets))

	local model = nn.Sequential()
		:add(nn.Sequencer(rm))
		:add(notes_offsets)

	return model

end 

return lstm