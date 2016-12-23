
local mnist = require 'mnist'

local mnist_loader = {}
mnist_loader.__index = mnist_loader

function mnist_loader.build(batch_size, sample)
	
	local self = {}
	setmetatable(self, mnist_loader)

	self.sample = sample or false
	self.batch_size = batch_size or 64
	self.raw_data = mnist.traindataset().data:double()

	self.data = self._preprocess(self.raw_data)
	self.index_batch = torch.randperm(self.data:size(1)):long():split(self.batch_size)
	self.n_batches = torch.floor(self.data:size(1)/self.batch_size)
	
	if not self.sample then
		self.current_batch = 0
		self.perm = torch.randperm(self.n_batches)
	end

	return self

end


function mnist_loader._preprocess(data)
	
	--standardise
	--return ((data-data:mean())/data:std()):view(-1, 28*28)
	
	--quantize
	return data:gt(10):view(-1, 28*28)
end


function mnist_loader:next_batch()
	local _nidx_
	if self.sample then
		_nidx_ = torch.random(self.n_batches)
	else
		self.current_batch = (self.current_batch % self.n_batches) + 1
		_nidx_ = self.perm[self.current_batch]
	end
	return self.data:index(1, self.index_batch[_nidx_])
end


function mnist_loader:getNumberOfBatches()
	return self.n_batches
end

return mnist_loader