
require 'nn'

local vae = {}

function vae.build_encoder(input_size, hidden_size, latent_size)
	
	local encoder = nn.Sequential()
		:add(nn.Linear(input_size, hidden_size))
		:add(nn.ReLU(true))

	local mean = nn.Linear(hidden_size, latent_size)
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


	decoder:add(nn.Linear(hidden_size, output_size))
	decoder:add(nn.Sigmoid(true))
	
	return decoder
end

return vae