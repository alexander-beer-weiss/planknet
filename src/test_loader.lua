
test_loader = {}
test_loader.__index = test_loader
setmetatable(test_loader, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function test_loader.new(opt)
  local self = setmetatable({},test_loader)
--   Just assuming the right parameters are here is lame and shitty
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end


function test_loader:loadNet()
	print '==> loading neural net'
	self.convNetObj = torch.load(self.netDatadir..'/NN.dat')
end

function test_loader:loadFileNames()
	print '==> loading test set file names'
	self.file_names = {}
	for file in paths.files(self.testingdir) do
		if string.match(file,'%.jpg$') then
			table.insert(self.file_names,file)
		end
	end
end


function test_loader:loadTestBatch(batch_num,batch_size)
	
	batch_size = batch_size or 20000  -- default value for batch_size
	
	local test_batch = {}
	
	start_index = (batch_num - 1) * batch_size + 1
	end_index = math.min(start_index + batch_size - 1, #self.file_names)
	
	print('==> loading batch ' .. batch_num .. ': ' .. end_index-start_index+1 .. ' images')

	for i = start_index, end_index do  -- read in names of all jpeg files; make table of these names
		local img = image.load(self.testingdir .. '/' .. self.file_names[i], 1, float)
		img = image.scale(img,self.height, self.width)
		img = img - torch.mean(img)
		img = img / torch.std(img)
		
		table.insert(test_batch, {self.file_names[i],img})
	end
	
	return test_batch
end

