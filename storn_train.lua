
require 'nn'
require 'nngraph'
require 'optim'
local display = require 'display'

require 'Sampler.lua'
require 'KLDCriterion'
require 'GaussianCriterion'
local storn = require 'storn'


local opt = {
	latent_size = 40,
	hidden_size = 400,
	n_layers = 2,
	dropout = true,
	batch_norm = true,
	stabilise = true,

	
	loader = 'MusicLoader',
	path = 'data/seqs',
	batch_size = 100,
	seq_length = 25,
	overlap = 25,
	sample_batch = true,

	max_epochs = 100,
	optim_alg = 'adam',
	optim_opt = {learningRate=2e-3},--, learningRateDecay=1e-2},
	seed = 129,
}

torch.manualSeed(opt.seed)

--loading and preparing batches
require(opt.loader)
local loader =  _G[opt.loader](opt)
opt.input_size = loader.data:size(3)
opt.output_size = loader.data:size(3)


--initializing encoder and decoder models
local encoder = storn.build_encoder(opt.input_size, opt.hidden_size,
	opt.latent_size, opt.n_layers, opt.dropout, opt.batch_norm, opt.stabilise)
local decoder = storn.build_decoder(opt.latent_size, opt.hidden_size,
	opt.output_size, opt.n_layers, opt.dropout, opt.batch_norm, opt.stabilise)


--regrouping encoder, Sampler and decoder into nn.gModule
local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})
local recons = decoder(z)
local model = nn.gModule({input}, {recons, mean, log_var})

--addressing all parameters
local params, grad_params = model:getParameters()

--initializing criterions
local KLDCriterion = nn.KLDCriterion()
local criterion = nn.BCECriterion()
criterion.sizeAverage = false




--training
local input = torch.Tensor(opt.seq_length, opt.batch_size, opt.input_size)

local feval = function(new_params)
	
	if params ~= new_params then
		params:copy(new_params)
	end

	model:zeroGradParameters()

	input:copy(loader:nextBatch())

	local recons, mean, log_var = unpack(model:forward(input))

	local KLDerr = KLDCriterion:forward(mean, log_var)
	local dKLD_dmu, dKlD_dlog_var = unpack(KLDCriterion:backward(mean, log_var))

	local err = criterion:forward(recons, input)
	local derr_dr = criterion:backward(recons, input)

	model:backward(input, {derr_dr, dKLD_dmu, dKlD_dlog_var})
	err = err + KLDerr

	grad_params:clamp(-5, 5)
	
	return err, grad_params
end



local lossPlot = {}
local gradNormPlot = {}
for epoch=1, opt.max_epochs do
	
	for iter=1, loader:getNumberOfBatches() do

		--training step
		_, fs = optim[opt.optim_alg](feval, params, opt.optim_opt)
		print('epoch '..epoch..', batch ' ..iter..'/'..loader:getNumberOfBatches()..': loss = '..fs[1])

		--evaluation
		lossPlot[#lossPlot+1] = {iter*epoch, fs[1]}
		display.plot(lossPlot, {win=23, title='Training Loss'})

		gradNormPlot[#gradNormPlot+1] = {iter*epoch, grad_params:norm()}
		display.plot(gradNormPlot, {win=25, title='Gradient Norm'})

		if iter%5==0 then
			decoder:evaluate()
			local seq = decoder:forward(torch.randn(1, opt.seq_length, opt.latent_size))
			torch.save('samples/sample_'..iter*epoch..'.t7', seq)
			decoder:training()
			display.image(seq, {win=24, title='first image generated'})
		end

	end

end