
require 'nn'
require 'nngraph'
require 'optim'

require 'Sampler.lua'
require 'KLDCriterion'
require 'GaussianCriterion'

local mnist_loader = require 'mnist_loader'
local vae = require 'vae'
local display = require 'display'


----------------  OPTIONS -------------------

local opt = {
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-3},--, weightDecay=1e-3},
	manual_seed = 11,
	batch_size = 128,
	input_size = 28*28,
	latent_size = 4,
	hidden_size = 400,
	max_epochs = 10,
}

---------------------------------------------

--torch.manualSeed(opt.manual_seed)

---------------- LOADING DATA ---------------
local loader = mnist_loader.build(opt.batch_size, false)
---------------------------------------------


-----------  MODEL AND CRITERION -------------


local encoder = vae.build_encoder(opt.input_size, opt.hidden_size, opt.latent_size)
local decoder = vae.build_decoder(opt.latent_size, opt.hidden_size, opt.input_size)

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)

local z = nn.Sampler()({mean, log_var})
local recons, recons_var = decoder(z):split(2)

local model = nn.gModule({input}, {recons, recons_var, mean, log_var})
local params, grad_params = model:getParameters()

local KLDCriterion = nn.KLDCriterion()

local criterion = nn.GaussianCriterion()
---------------------------------------------


------------------- TRAINING ----------------
local input = torch.Tensor(opt.batch_size, opt.input_size)

local feval = function(new_params)
	if params ~= new_params then
		params:copy(new_params)
	end

	grad_params:zero()

	input:copy(loader:next_batch())

	local recons, recons_var, mean, log_var = unpack(model:forward(input))
	
	--print(mean)
	local KLDerr = KLDCriterion:forward(mean, log_var)
	print('KLDerr')
	print(KLDerr)
	local dKLD_dmu, dKlD_dlog_var = unpack(KLDCriterion:backward(mean, log_var))

	--print(recons)
	local err = criterion:forward({recons, recons_var}, input)
	print('err')
	print(err)
	local derr_dr, derr_drvar = unpack(criterion:backward({recons, recons_var}, input))

	model:backward(input, {derr_dr, derr_drvar, dKLD_dmu, dKlD_dlog_var})
	err = err + KLDerr

	return err, grad_params
end

local lossPlot = {}
for epoch=1, opt.max_epochs do
	for iter=1, loader:getNumberOfBatches() do
		_, fs = optim[opt.optim_alg](feval, params, opt.optim_opt)
		lossPlot[#lossPlot+1] = {iter*epoch, fs[1]}
		display.plot(lossPlot, {win=23, title='Train Loss Plot'})
		if iter%5==0 then
			decoder:evaluate()
			im, _ = unpack(decoder:forward(torch.randn(opt.latent_size)))
			decoder:training()
			display.image(im:view(28,28), {win=24, title='first image generated'})
		end
		print('epoch '..epoch..': loss = '..fs[1])
	end
end

---------------------------------------------
