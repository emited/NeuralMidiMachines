
require 'nn'
require 'nngraph'
require 'optim'
local display = require 'display'

require 'Sampler.lua'
require 'KLDCriterion'
require 'GaussianCriterion'
require 'NoteLoader'
local utils = require 'utils'
local vrae = require 'vrae_gru_new'


local opt = {
	latent_size = 100,
	hidden_size = 400,
	n_layers = 1,
	dropout = false,
	
	path = 'rythm/data/seqs_rythm',
	batch_size = 1,
	seq_length = 15,
	overlap = 15,
	sample_batch = true,

	max_epochs = 10000,
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-3, learningRateDecay=1e-3},
	seed = 123,
}

--torch.manualSeed(opt.seed)

--loading and preparing batches
local loader = NoteLoader(opt)
opt.input_size = 128
print('okokok')

--initializing encoder and decoder models
local encoder = vrae.build_encoder(opt.input_size, opt.hidden_size,
	opt.latent_size, opt.n_layers, opt.dropout)

local map_z_to_hidden = nn.Sequential()
	:add(nn.Linear(opt.latent_size, opt.hidden_size))
	:add(nn.Tanh())

local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})

local h0 = map_z_to_hidden(z)
local encoder_sampler = nn.gModule({input}, {h0, mean, log_var})


local decoder = vrae.build_decoder(opt.input_size, opt.hidden_size,
	opt.input_size, opt.n_layers, opt.dropout)

local model = nn.Container()
	:add(encoder_sampler)
	:add(decoder)

local params, grad_params = model:getParameters()

local KLD = nn.KLDCriterion()

local criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())


local init_grad_h = torch.Tensor()

local feval = function(new_params)

	if params ~= new_params then
		params:copy(new_params)
	end 

	model:zeroGradParameters()

	local input_encoder = loader:nextBatch():clone()
	print(input_encoder:size())
	local init_h, mean, log_var = unpack(encoder_sampler:forward(input_encoder))

	local KLDerr = KLD:forward(mean, log_var)
	local dKLD_dmu, dKlD_dlog_var = unpack(KLD:backward(mean, log_var))

	decoder.set_init_h(init_h)
	local recons = decoder:forward(input_decoder)
	local err = criterion:forward(recons, target_decoder)

	local derr_dr = criterion:backward(recons, target_decoder)
	local dd_dz = decoder:backward(input_decoder, derr_dr)

	init_grad_h = decoder.copy_init_grad_h(init_grad_h)
	encoder_sampler:backward(input_encoder, {init_grad_h,  dKLD_dmu, dKlD_dlog_var})

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
		gradNormPlot[#gradNormPlot+1] = {iter*epoch, grad_params:norm()}

		if iter%5 == 0 then
			decoder:evaluate()
			local z = torch.randn(opt.latent_size)
			local init_h = map_z_to_hidden:forward(z)
			local sample = decoder.sample(init_h, opt.seq_length)
			display.image(sample, {win=24, title='Generation from rand sample'})
			decoder:training()
		end

		display.plot(lossPlot, {win=23, title='Training Loss'})
		display.plot(gradNormPlot, {win=25, title='Gradient Norm'})
	end
end