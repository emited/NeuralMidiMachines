
require 'paths'
require 'Loader'


local MusicLoader = torch.class('MusicLoader', 'Loader')

function MusicLoader:setData(opt)

	local seq_length = opt.seq_length or 50
	local overlap = opt.overlap or 25

	local fns = getFilenames(opt.path)

	local seqs = {}
	for i, fn in ipairs(fns) do
		 local seq = parseSeqFile(fn)
		 for t = 1, seq.notes:size(1)-seq_length, overlap do
		 	seqs[#seqs+1] = seq.notes:sub(t, t + seq_length-1)
		 end
	end

	local data = torch.DoubleTensor(#seqs, seq_length, 128)
	for i, seq in ipairs(seqs) do
		data[i] = seq
	end

	self.batch_first = true
	self.seq_length = seq_length
	self.overlap = overlap

	print('loaded and created '..#seqs..' seqs sucessfully.')
	return data

end

function MusicLoader:preprocess(data)
	return data:view(data:size(1), -1)
end


function getFilenames(path)
	local curr_path = paths.cwd()..'/'..path
	local fns = {}
	for fn in paths.files(curr_path) do
	    if fn:find('seq' .. '$') then
	    	fns[#fns+1] = paths.concat(curr_path, fn)
	    end
	end
	return fns
end


function parseSeqFile(fn)

	--parse seq file
	local file = io.open(fn)
	local seq = {}
	seq.notes = {}
	seq.offsets = {}
	for line in file:lines() do
		local offset, notes = unpack(line:split(':'))
		local pos = #seq.notes + 1
		seq.offsets[pos] = tonumber(offset)
		seq.notes[pos] = {}
		for _, note in pairs(notes:split(',')) do
			local pos_note = #seq.notes[pos] + 1
			seq.notes[pos][pos_note] = tonumber(note)
		end
	end

	--convert to tensor
	local tseq = {}
	tseq.offsets = torch.Tensor(seq.offsets)
	tseq.notes = torch.ByteTensor(#seq.notes, 128):zero()
	for t, notes in ipairs(seq.notes) do
		for _, note in ipairs(notes) do
			tseq.notes[{t, note+1}] = 1
		end
	end

	return tseq

end