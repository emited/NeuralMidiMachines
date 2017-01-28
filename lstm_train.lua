
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
local lstm = require 'lstm'
local display = require 'display'


local opt = {
	hidden_size = 300,
	dropout = false,
	n_layers = 2,
	stabilise = true,
	batch_norm = true,
	
	loader = 'MusicLoader',
	path = 'data/seqs_transposed',
	batch_size = 128,
	seq_length = 25,
	overlap = 4,
	sample_batch = true,

	max_epochs = 100,
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-2, learningRateDecay=1e-3},
	seed = 133,
}

torch.manualSeed(opt.seed)

--loading and preparing batches
require(opt.loader)
local loader =  _G[opt.loader](opt)
opt.input_size = loader.data:size(3)
opt.output_size = opt.input_size

local model = lstm.build(opt.input_size, opt.hidden_size, opt.output_size,
	opt.n_layers, opt.dropout, opt.batch_norm, opt.stabilise)
local params, grad_params = model:getParameters()

local criterion = nn.SequencerCriterion(nn.BCECriterion())


local input = torch.Tensor(opt.seq_length, opt.batch_size, opt.input_size):zero()
local target = torch.Tensor(opt.seq_length, opt.batch_size, opt.input_size):zero()

local feval = function(new_params)
	
	if params ~= new_params then
		params:copy(new_params)
	end

	model:zeroGradParameters()

	local sequence = loader:nextBatch()

	input:sub(2, -1):copy(sequence:sub(1, -2))
	target:copy(sequence)

	local output = model:forward(input)
	local err = criterion:forward(output, target)
	local delta = criterion:backward(output, target)
	model:backward(output, delta)

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
		display.plot(lossPlot, {win=29, title='Training Loss'})

		gradNormPlot[#gradNormPlot+1] = {iter*epoch, grad_params:norm()}
		display.plot(gradNormPlot, {win=28, title='Gradient Norm'})

		--hallucination for evaluation
		if iter%5==0 then
			model:evaluate()
			local amaxs = {torch.Tensor():resize(1,opt.input_size):zero()}
			local outputs = {}
			for t=1, opt.seq_length do
				local output = model:forward(amaxs)[#amaxs]
				table.insert(outputs, output)
				table.insert(amaxs, output:eq(output:max()):double())
			end
			model:training()
			local outputs = torch.cat(outputs, 1)
			local amaxs = torch.cat(amaxs, 1)
			torch.save('samples/sample_'..(epoch-1)*loader:getNumberOfBatches()+iter..'.t7', amaxs)
			print('saving to '..'samples/sample_'..(epoch-1)*loader:getNumberOfBatches()+iter..'.t7')
			display.image(outputs, {win=27, title='Generation: Outputs'})
			display.image(amaxs, {win=30, title='Generation: Argmax of Outputs'})
		end

	end

end