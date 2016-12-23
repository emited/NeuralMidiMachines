
require 'nn'

local vae = {}

function vae.build_encoder(input_size, hidden_size, latent_size)
	
	local encoder = nn.Sequential()
		:add(nn.Linear(input_size, hidden_size))
		:add(nn.ReLU(true))
		:add(nn.Dropout(0.5))
		--:add(nn.Linear(hidden_size, hidden_size))


	local mean = nn.Linear(hidden_size, latent_size)

	--constrained to be a diagonal matrix
	local logvar = nn.Linear(hidden_size, latent_size)

	encoder:add(
		nn.ConcatTable()
			:add(mean)
			:add(logvar))

	return encoder
end

function vae.build_decoder(latent_size, hidden_size, output_size)

	local decoder = nn.Sequential()
		:add(nn.Linear(latent_size, hidden_size))
		:add(nn.ReLU(true))
		:add(nn.Linear(hidden_size, hidden_size))
		:add(nn.ReLU(true))


	local mean = nn.Linear(hidden_size, output_size)
	local logvar = nn.Linear(hidden_size, output_size)
	
	decoder:add(
		nn.ConcatTable()
			:add(mean)
			:add(logvar))

	return decoder
end

return vae