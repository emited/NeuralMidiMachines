
require 'paths'
require 'Loader'


local seqLoader = torch.class('seqLoader', 'Loader')

function seqLoader:setData(opt)

	self.seq_length = opt.seq_length or 50
	self.overlap = opt.overlap or 25

	local fns = getFilenames('data/seqs')

	local seqs = {}
	for i, fn in ipairs(fns) do
		 local seq = parseSeqFile(fn)
		 for t = 1, seq.notes:size(1)-seq_length, overlap do
		 	seqs[#seqs+1] = seq.notes:sub(t, t + seq_length-1)
		 end
	end

	local data = torch.ByteTensor(#seqs, seq_length, 128)
	for i, seq in ipairs(seqs) do
		data[i] = seq
	end

	return data
end


function getFilenames(path)
	curr_path = paths.cwd()..'/'..path
	fns = {}
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
	tseq = {}
	tseq.offsets = torch.Tensor(seq.offsets)
	tseq.notes = torch.ByteTensor(#seq.notes, 128):zero()
	for t, notes in ipairs(seq.notes) do
		for _, note in ipairs(notes) do
			tseq.notes[{t, note+1}] = 1
		end
	end

	return tseq

end