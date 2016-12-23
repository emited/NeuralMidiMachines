
local music_loader = {}
music_loader.__index = music_loader



function music_loader.build(sample)
	local self = {}
	setmetatable(self, music_loader)

	self.sample = sample or false

	return self
end



function music_loader:next_batch()
	return
end


return music_loader