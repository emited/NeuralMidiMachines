
local mnist = require 'mnist'
require 'Loader'

local MnistLoader = torch.class('MnistLoader', 'Loader')

function MnistLoader:setData()
	data = mnist.traindataset().data:double()
	return data:gt(1):view(-1, 28*28)
end