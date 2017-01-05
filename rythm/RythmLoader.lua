
require 'paths'

local RythmLoader = torch.class('RythmLoader')


function RythmLoader:__init(opt)
	
	self.sample = opt.sample or false
	self.batch_size = opt.batch_size or 64
	self.batch_first = opt.batch_first or false

	self.notes, self.offsets = self:setData(opt)

	self.index_batch = torch.randperm(self.notes:size(1)):long():split(self.batch_size)
	self.n_batches = torch.floor(self.notes:size(1)/self.batch_size)

	if not self.sample then
		self.current_batch = 0
		self.perm = torch.randperm(self.n_batches)
	end

end


function RythmLoader:nextBatch()
	
	local _nidx_
	if self.sample then
		_nidx_ = torch.random(self.n_batches)
	else
		self.current_batch = (self.current_batch % self:getNumberOfBatches()) + 1
		_nidx_ = self.perm[self.current_batch]
	end

	local notes, offsets
	if self.batch_first then
		notes = self.notes:index(1, self.index_batch[_nidx_]):transpose(1, 2)
		offsets = self.offsets:index(1, self.index_batch[_nidx_]):transpose(1, 2)
	else
		notes = self.notes:index(1, self.index_batch[_nidx_])
		offsets = self.offsets:index(1, self.index_batch[_nidx_])
	end
	
	return notes, offsets
end


function RythmLoader:getNumberOfBatches()
	return self.n_batches
end


function RythmLoader:setData(opt)

	self.idx2offset = require 'data.notes'
	
	self.offset2idx = {}
	for i, v in ipairs(self.idx2offset) do
		self.offset2idx[v] = i
	end

	local seq_length = opt.seq_length or 50
	local overlap = opt.overlap or 25

	local fns = self:getFilenames(opt.path)

	local all_notes, all_offsets = {}, {}
	for i, fn in ipairs(fns) do

		local notes, offsets = self:parseSeqFile(fn)

		for t = 1, notes:size(1)-seq_length, overlap do
			all_notes[#all_notes+1] = notes:sub(t, t + seq_length-1)
			all_offsets[#all_offsets+1] = offsets:sub(t, t + seq_length-1)
		end

	end
	local batch_offsets = torch.ByteTensor(#all_offsets, seq_length, #self.idx2offset):zero()
	local batch_notes = torch.ByteTensor(#all_notes, seq_length, 128):zero()

	for i, offsets in ipairs(all_offsets) do
		batch_offsets[i] = offsets
	end

	for i, notes in ipairs(all_notes) do
		batch_notes[i] = notes
	end

	--local notes = torch.DoubleTensor(#b_notes, seq_length, 128)
	--local offsets = torch.DoubleTensor(#b_notes, seq_length, 128)
	--for i,  in ipairs(b_notes) do
	--	notes[i] = seq
	--end

	self.batch_first = true
	self.seq_length = seq_length
	self.overlap = overlap

	print('loaded and created '..#all_notes..' seqs sucessfully.')
	return batch_notes, batch_offsets

end


function RythmLoader:getFilenames(path)
	local curr_path = paths.cwd()..'/'..path
	local fns = {}
	for fn in paths.files(curr_path) do
	    if fn:find('seq' .. '$') then
	    	fns[#fns+1] = paths.concat(curr_path, fn)
	    end
	end
	return fns
end


function RythmLoader:parseSeqFile(fn)

	--parse seq file
	local file = io.open(fn)
	local seq = {}
	local notes, offsets = {}, {}
	local i = 1
	for line in file:lines() do
		local offset, note = unpack(line:split(':'))
		offsets[i] = tonumber(offset)
		notes[i] = tonumber(note)
		i = i + 1
	end

	local t_notes = torch.ByteTensor(#notes, 128):zero()
	for t, note in ipairs(notes) do
		t_notes[{t, note+1}] = 1
	end

	local t_offsets = torch.Tensor(#offsets, #self.idx2offset):zero()
	for t, offset in ipairs(offsets) do
		local k = self.offset2idx[offset]
		t_offsets[t][k] = 1
	end

	return t_notes, t_offsets

end