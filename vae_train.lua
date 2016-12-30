
require 'nn'
require 'nngraph'
require 'optim'

require 'Sampler.lua'
require 'KLDCriterion'
require 'GaussianCriterion'

local vae = require 'vae'
local display = require 'display'


----------------  OPTIONS -------------------

local opt = {
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-3},--, weightDecay=1e-3},
	loader='MusicLoader',
	path = 'data/seqs_transposed',
	seq_length = 25,
	manual_seed = 129,
	batch_size = 16,
	hidden_size = 800,
	latent_size = 20,
	max_epochs = 100,
}

---------------------------------------------

--torch.manualSeed(opt.manual_seed)

---------------- LOADING DATA ---------------
require(opt.loader)
local loader =  _G[opt.loader](opt)
opt.input_size = loader.data:size(2)
---------------------------------------------


-----------  MODEL AND CRITERION -------------

local encoder = vae.build_encoder(opt.input_size, opt.hidden_size, opt.latent_size)
local decoder = vae.build_decoder(opt.latent_size, opt.hidden_size, opt.input_size)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)

local z = nn.Sampler()({mean, log_var})
local recons = decoder(z)

local model = nn.gModule({input}, {recons, mean, log_var})
local params, grad_params = model:getParameters()

local KLD = nn.KLDCriterion()

local criterion = nn.BCECriterion()
criterion.sizeAverage = false
---------------------------------------------


------------------- TRAINING ----------------
local input = torch.Tensor(opt.batch_size, opt.input_size)

local feval = function(new_params)
	if params ~= new_params then
		params:copy(new_params)
	end

	model:zeroGradParameters()

	input:copy(loader:nextBatch())

	local recons, mean, log_var = unpack(model:forward(input))
	
	local KLDerr = KLD:forward(mean, log_var)
	local dKLD_dmu, dKlD_dlog_var = unpack(KLD:backward(mean, log_var))

	local err = criterion:forward(recons, input)
	local derr_dr = criterion:backward(recons, input)

	model:backward(input, {derr_dr, dKLD_dmu, dKlD_dlog_var})
	
	err = err + KLDerr

	return err, grad_params
end

local lossPlot = {}
for epoch=1, opt.max_epochs do
	for iter=1, loader:getNumberOfBatches() do
		_, fs = optim[opt.optim_alg](feval, params, opt.optim_opt)
		lossPlot[#lossPlot+1] = {iter*epoch, fs[1]}
		display.plot(lossPlot, {win=43, title='Train Loss Plot'})
		if iter%5==0 then
			decoder:evaluate()
			im = decoder:forward(torch.randn(opt.latent_size))
			decoder:training()
			display.image(im:view(loader.seq_length, 128), {win=44, title='first image generated'})
		end
		print('epoch '..epoch..': loss = '..fs[1])
	end
end

---------------------------------------------
