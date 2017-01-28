
require 'nn'
require 'nngraph'
require 'optim'
require 'rnn'
local lstm = require 'lstm2'
local display = require 'display'



function one_hot(tensor, max_val)
	--print(tensor)
	local oh = torch.ByteTensor(tensor:size(1), max_val):zero()
	for i=1, tensor:size(1) do
		oh[i][tensor[i]] = 1
	end
	return oh
end


local opt = {
	hidden_size = 600,
	dropout = false,
	n_layers = 1,
	stabilise = true,
	batch_norm = true,
	
	loader = 'RythmLoader',
	path = 'data/seqs_rythm',
	batch_size = 64,
	seq_length = 15,
	overlap = 15,
	sample_batch = false,

	max_epochs = 100,
	optim_alg = 'adam',
	optim_opt = {learningRate=3e-2, learningRateDecay=1e-4},
	seed = 133,
}

--torch.manualSeed(opt.seed)

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
	model:forget()

	notes, offsets = loader:nextBatch()

	out_notes, out_offsets = unpack(model:forward({notes.input, offsets.input}))

	local note_err = note_criterion:forward(out_notes, notes.target)
	local offset_err = offset_criterion:forward(out_offsets, offsets.target)

	local note_delta = note_criterion:backward(out_notes, notes.target)
	local offset_delta = offset_criterion:backward(out_offsets, offsets.target)

	model:backward({notes.input, offsets.input}, {note_delta, offset_delta})

	--grad_params:clamp(-10, 10) --clipping gradients

	local err = note_err + offset_err

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
		if iter%10==0 then
			
			model:evaluate()
			model:forget()
			
			local gen_notes = torch.LongTensor({{loader.go_note_token}})
			local gen_offsets = torch.LongTensor({{loader.go_offset_token}})

			local prob_notes = nil
			local prob_offsets = nil

			local cond_notes = notes.input[{{}, 1}]:clone():long()
			local cond_offsets = offsets.input[{{}, 1}]:clone():long()
			gen_notes = cond_notes:sub(1, 10)
			gen_notes = gen_notes:view(-1, 1)
			gen_offsets = cond_offsets:sub(1, 10)
			gen_offsets = gen_offsets:view(-1, 1)
			
			--for i=1, 10 do
			--	model:forward({cond_notes[]})
			--end

			for t=1, opt.seq_length do

				local prob_note, prob_offset = unpack(model:forward({gen_notes, gen_offsets}))
				
				_, gen_note = prob_note[-1]:max(2)
				_, gen_offset = prob_offset[-1]:max(2)
				gen_notes = torch.cat({gen_notes, gen_note}, 1)
				gen_offsets = torch.cat({gen_offsets, gen_offset}, 1)

				if t==1 then
					prob_notes = prob_note[-1]:clone()
					prob_offsets = prob_offset[-1]:clone()
				else
					prob_notes = torch.cat({prob_notes, prob_note[-1]:clone()}, 1)
					prob_offsets = torch.cat({prob_offsets, prob_offset[-1]:clone()}, 1)
				end

			end
			model:training()
			--torch.save((epoch-1)*loader:getNumberOfBatches()+iter..'.t7', gen_notes)
			--print('saving to '..'samples/sample_'..(epoch-1)*loader:getNumberOfBatches()+iter..'.t7')
			display.image(prob_notes, {win=47, title='Generation: Note Outputs'})
			display.image(prob_offsets, {win=48, title='Generatation: Offset Outputs'})
			display.image(one_hot(gen_notes:squeeze(), opt.note_size+1), {win=40, title='Generation: Argmax of Notes'})
			display.image(one_hot(notes.input[{{}, 1}]:squeeze(), opt.note_size+1), {win=52, title='Input Notes'})
			display.image(one_hot(offsets.input[{{}, 1}]:squeeze(), opt.note_size+1), {win=58, title='Input Offsets'})
			display.image(out_notes[{{}, 1}], {win=53, title='Train Notes'})
			display.image(out_offsets[{{}, 1}], {win=54, title='Train Offsets'})
			display.image(one_hot(gen_offsets:squeeze(), opt.offset_size+1), {win=42, title='Generation: Argmax of Offsets'})
		end

	end

end