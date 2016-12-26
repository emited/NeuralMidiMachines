
require 'nn'
require 'nngraph'
require 'optim'
local display = require 'display'

require 'Sampler.lua'
require 'KLDCriterion'
require 'GaussianCriterion'
local storn = require 'storn'


local opt = {
	input_size = 100,
	latent_size = 30,
	hidden_size = 100,
	output_size = 100,
	dropout = false,
	n_layers = 10,
	
	loader = 'mnistLoader',
	filename = '',
	batch_size = 6,
	seq_length = 7,
	sample_batch = false,

	max_epochs = 10,
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-4, },
}


--loading and preparing batches
require(opt.loader)
local loader =  _G[opt.loader](opt)

--initializing model
--initializing encoder and decoder models
local encoder = storn.build_encoder(opt.input_size, opt.hidden_size,
	opt.latent_size, opt.n_layers, opt.dropout)
local decoder = storn.build_decoder(opt.latent_size, opt.hidden_size,
	opt.output_size, opt.n_layers, opt.dropout)

local inputs = torch.rand(opt.seq_length, opt.batch_size, opt.input_size)
--local output = encoder:forward(inputs)
--print(output)

--regrouping encoder, Sampler and decoder into nn.gModule
local input = nn.Identity()()
local mean, log_var = encoder(input):split(2)
local z = nn.Sampler()({mean, log_var})
local recons, recons_var = decoder(z):split(2)
local model = nn.gModule({input}, {recons, recons_var, mean, log_var})

--addressing all parameters
local params, grad_params = model:getParameters()

--initializing criterions
local KLDCriterion = nn.KLDCriterion()
local criterion = nn.GaussianCriterion()




--training
local input = torch.Tensor(opt.seq_length, opt.batch_size, opt.input_size)

local feval = function(new_params)
	
	if params ~= new_params then
		params:copy(new_params)
	end

	grad_params:zero()

	input:copy(torch.rand(opt.seq_length, opt.batch_size, opt.input_size))
	--input:copy(loader:next_batch())

	local recons, recons_var, mean, log_var = unpack(model:forward(input))
	
	local KLDerr = KLDCriterion:forward(mean, log_var)
	local dKLD_dmu, dKlD_dlog_var = unpack(KLDCriterion:backward(mean, log_var))

	local err = criterion:forward({recons, recons_var}, input)
	local derr_dr, derr_drvar = unpack(criterion:backward({recons, recons_var}, input))

	model:backward(input, {derr_dr, derr_drvar, dKLD_dmu, dKlD_dlog_var})
	err = err + KLDerr

	return err, grad_params
end



local lossPlot = {}
for epoch=1, opt.max_epochs do
	for iter=1, loader:getNumberOfBatches() do
		_, fs = optim[opt.optim_alg](feval, params, opt.optim_opt)
		print('epoch '..epoch..': loss = '..fs[1])
	end
end

print(decoder:forward(output[1]))
