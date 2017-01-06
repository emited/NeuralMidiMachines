
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
local lstm = require 'lstm'
local display = require 'display'


local opt = {
	hidden_size = 100,
	dropout = false,
	n_layers = 1,
	stabilise = false,
	batch_norm = true,
	
	loader = 'RythmLoader',
	path = 'data/seqs_rythm',
	batch_size = 13,
	seq_length = 25,
	overlap = 25,
	sample_batch = false,

	max_epochs = 100,
	optim_alg = 'adam',
	optim_opt = {learningRate=1e-2, learningRateDecay=1e-3},
	seed = 131,
}

torch.manualSeed(opt.seed)

--loading and preparing batches
require(opt.loader)
local loader =  RythmLoader(opt)
opt.note_size = 128
opt.offset_size = 76

opt.output_size = opt.input_size

local model = lstm.build(opt.note_size, opt.offset_size, opt.hidden_size,
	opt.n_layers, opt.dropout, opt.batch_norm, opt.stabilise)
local params, grad_params = model:getParameters()

local note_criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())
local offset_criterion = nn.SequencerCriterion(nn.CrossEntropyCriterion())


local feval = function(new_params)
	
	if params ~= new_params then
		params:copy(new_params)
	end

	model:zeroGradParameters()

	local notes, offsets = loader:nextBatch()
	print(notes)
	print(offsets)
	require 'dpnn'
	require 'rnn'
	print(notes.input)
	print(nn.Sequencer(nn.OneHot(128)):forward(notes.input))

	local out_notes, out_offsets = unpack(model:forward({notes.input, offsets.input}))

	local note_err = note_criterion:forward(out_notes, notes.target)
	local offset_err = offset_criterion:forward(out_offsets, notes.offsets)

	--model:backward(output, delta)

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
			local outputs = {torch.Tensor():resize(1,opt.input_size):zero()}
			for t=1, opt.seq_length do
				output = model:forward(outputs)[#outputs]
				table.insert(outputs, output:eq(output:max()):double())
			end
			model:training()
			local output = torch.cat(outputs, 1)
			model:training()
			display.image(output, {win=27, title='first image generated'})
		end

	end

end