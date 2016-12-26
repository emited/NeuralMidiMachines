
local mnist = require 'mnist'
require 'Loader'

local mnistLoader = torch.class('mnistLoader', 'Loader')

function mnistLoader:setData()
	data = mnist.traindataset().data:double()
	return data:gt(10):view(-1, 28*28)
end