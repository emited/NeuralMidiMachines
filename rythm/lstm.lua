
require 'nn'
require 'rnn'
require 'dpnn'

local lstm = {}

function lstm.build(note_size, offset_size, hidden_size, n_layers, dropout, batch_norm, stabilise)

	local n_layers = n_layers or 1
	local batch_norm = batch_norm or false
	local stabilise = stabilise or false
	local dropout = dropout or false


	local note2h = nn.Sequential()
		:add(nn.LookupTable(note_size+1, hidden_size))

	local offset2h = nn.Sequential()
		:add(nn.LookupTable(offset_size+1, hidden_size))

	local model = nn.Sequential()

	model:add(
		nn.ParallelTable()
			:add(nn.Sequencer(note2h))
			:add(nn.Sequencer(offset2h))
	)

	model:add(nn.CAddTable())

	local rm = nn.Sequential()

	for layer=1, n_layers do 

		--adding lstm layer
		nn.FastLSTM.bn = batch_norm
		rm:add(nn.FastLSTM(hidden_size, hidden_size))

		--adding stabilising regularization
		if stabilise then
			rm:add(nn.NormStabilizer())
		end

		--adding dropout layer
		if dropout then
			rm:add(nn.Dropout(0.5))
		end

	end

	model:add(nn.Sequencer(rm))

	local notes = nn.Sequential()
		:add(nn.Linear(hidden_size, note_size))
		:add(nn.SoftMax())

	local offsets = nn.Sequential()
		:add(nn.Linear(hidden_size, offset_size))
		:add(nn.SoftMax())

	local notes_offsets = nn.ConcatTable()
		:add(nn.Sequencer(notes))
		:add(nn.Sequencer(offsets))

	model:add(nn.Sequential()
		:add(notes_offsets))

	return model:remember('both')

end 

return lstm