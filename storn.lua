
require 'nn'
require 'rnn'

local storn = {}

function storn.build_encoder(input_size, hidden_size, latent_size, n_layers, dropout)

	local n_layers = n_layers or 1

	local encoder = nn.Sequential()
	
	for layer=1, n_layers do 

		--adding lstm layer
		local seq_lstm
		if layer == 1 then
		 seq_lstm = nn.SeqLSTM(input_size, hidden_size)
		else
		 seq_lstm = nn.SeqLSTM(hidden_size, hidden_size)
		end
		encoder:add(seq_lstm)

		--adding dropout layer
		if not dropout then
			encoder:add(nn.Dropout(dropout))
		end

	end

	--adding mean and logvar output
	local mean = nn.Linear(hidden_size, latent_size)
	local logvar = nn.Linear(hidden_size, latent_size)
	local means_logvars = nn.ConcatTable()
		:add(nn.Sequencer(mean))
		:add(nn.Sequencer(logvar))

	encoder:add(means_logvars)

	return encoder

end 



function storn.build_decoder(latent_size, hidden_size, output_size, n_layers, dropout)

	local n_layers = n_layers or 1

	local decoder = nn.Sequential()

	for layer=1, n_layers do

		--adding lstm layer
		local seq_lstm
		if layer == 1 then
		 seq_lstm = nn.SeqLSTM(latent_size, hidden_size)
		else
		 seq_lstm = nn.SeqLSTM(hidden_size, hidden_size)
		end
		decoder:add(seq_lstm)

		--adding dropout layer
		if not dropout then
			decoder:add(nn.Dropout(dropout))
		end
	end

	--adding mean and logvar output
	local mean = nn.Linear(hidden_size, output_size)
	local logvar = nn.Linear(hidden_size, output_size)
	local means_logvars = nn.ConcatTable()
		:add(nn.Sequencer(mean))
		:add(nn.Sequencer(logvar))

	decoder:add(means_logvars)

	return decoder

end

return storn