
require 'paths'

local RythmLoader = torch.class('RythmLoader')


function RythmLoader:__init(opt)
	
	self.path = opt.path
	self.sample = opt.sample or false
	self.batch_size = opt.batch_size or 64
	self.seq_length = opt.seq_length or 50
	self.overlap = opt.overlap or 25
	self.note_size = 128

	self.idx2offset = require 'data.notes'
	
	self.offset2idx = {}
	for i, v in ipairs(self.idx2offset) do
		self.offset2idx[v] = i
	end
	self.offset_size = #self.idx2offset

	self.notes, self.offsets = self:setData()

	self.index_batch = torch.randperm(self.notes:size(1)):long():split(self.batch_size)
	self.n_batches = torch.floor(self.notes:size(1)/self.batch_size)

	if not self.sample then
		self.current_batch = 0
		self.perm = torch.randperm(self.n_batches)
	end

	self.bnotes = {}
	self.bnotes.input = torch.Tensor(self.seq_length, self.batch_size):zero()
	self.bnotes.target = torch.Tensor(self.seq_length, self.batch_size)
	
	self.boffsets = {}
	self.boffsets.input = torch.Tensor(self.seq_length, self.batch_size):zero()
	self.boffsets.target = torch.Tensor(self.seq_length, self.batch_size)

end


function RythmLoader:nextBatch()
	
	local _nidx_
	if self.sample then
		_nidx_ = torch.random(self.n_batches)
	else
		self.current_batch = (self.current_batch % self:getNumberOfBatches()) + 1
		_nidx_ = self.perm[self.current_batch]
	end

	local all_notes = self.notes:index(1, self.index_batch[_nidx_]):transpose(1, 2)
	local all_offsets = self.offsets:index(1, self.index_batch[_nidx_]):transpose(1, 2)

	self.bnotes.input:sub(2, -1):copy(all_notes:sub(1, -2))
	self.boffsets.input:sub(2, -1):copy(all_offsets:sub(1, -2))
	
	self.bnotes.target:copy(all_notes)
	self.boffsets.target:copy(all_offsets)
	
	return self.bnotes, self.boffsets
end


function RythmLoader:getNumberOfBatches()
	return self.n_batches
end


function RythmLoader:setData(opt)

	local fns = self:getFilenames(self.path)

	local all_notes, all_offsets = {}, {}
	for i, fn in ipairs(fns) do

		local notes, offsets = self:parseSeqFile(fn)

		for t = 1, notes:size(1)-self.seq_length, self.overlap do
			all_notes[#all_notes+1] = notes:sub(t, t + self.seq_length-1):view(1, -1)
			all_offsets[#all_offsets+1] = offsets:sub(t, t + self.seq_length-1):view(1, -1)
		end

	end

	local batch_notes = torch.cat(all_notes, 1)
	local batch_offsets = torch.cat(all_offsets, 1)

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
		offsets[i] = self.offset2idx[tonumber(offset)]
		notes[i] = tonumber(note)
		i = i + 1
	end

	--FOR NON ONE HOT
	--local t_notes = torch.ByteTensor(#notes, self.note_size):zero()
	--for t, note in ipairs(notes) do
	--	t_notes[{t, note+1}] = 1
	--end
	--
	--local t_offsets = torch.Tensor(#offsets, self.offset_size):zero()
	--for t, offset in ipairs(offsets) do
	--	local k = self.offset2idx[offset]
	--	t_offsets[t][k] = 1
	--end

	return torch.ByteTensor(notes), torch.ByteTensor(offsets)

end