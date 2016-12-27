
--[[
Abstract class for Loader
Requires overloading setData function,
has to return a Tensor of size n_batches x data_size
]]

local Loader = torch.class('Loader')

function Loader:__init(opt)
	
	self.sample = opt.sample or false
	self.batch_size = opt.batch_size or 64
	self.batch_first = opt.batch_first or false

	self.data = self:setData(opt)

	self.index_batch = torch.randperm(self.data:size(1)):long():split(self.batch_size)
	self.n_batches = torch.floor(self.data:size(1)/self.batch_size)

	if not self.sample then
		self.current_batch = 0
		self.perm = torch.randperm(self.n_batches)
	end

end

function Loader:setData(opt)
	assert(false)
end

function Loader:nextBatch()
	local _nidx_
	if self.sample then
		_nidx_ = torch.random(self.n_batches)
	else
		self.current_batch = (self.current_batch % self:getNumberOfBatches()) + 1
		_nidx_ = self.perm[self.current_batch]
	end
	if self.batch_first then
		return self.data:index(1, self.index_batch[_nidx_]):transpose(1, 2)
	else
		return self.data:index(1, self.index_batch[_nidx_])
	end
end


function Loader:getNumberOfBatches()
	return self.n_batches
end