
--[[
image,acantharia_protist_big_center,...,unknown_unclassified
1.jpg,0.00826446,...,0.00826446
10.jpg,0.00826446,...,0.00826446
--]]

require 'paths'  -- read OS directory structure


csv_writer = {}
csv_writer.__index = csv_writer
setmetatable(csv_writer, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function csv_writer.new(opt)
	local self = setmetatable({},csv_writer)
	-- Just assuming the right parameters are here is lame and shitty
	for key, value in pairs(opt) do
		self[key] = value
	end
	return self
end

function csv_writer:writeToCSV(lines, header)
	
	if not paths.dir(self.outputdir) then
	  print('==> creating directory '..self.outputdir)
	  paths.mkdir(self.outputdir)
	end

	print '==> writing to csv'
	local file_stream = io.open(self.outputdir..'/planknet.csv','w')

	if header then
		for name, line in pairs(header) do
			file_stream:write(name)
			for _, value in ipairs(line) do
				file_stream:write(',', value)
			end
			file_stream:write('\n')
		end
	end  

	for name,line in pairs(lines) do
		file_stream:write(name)
		for i = 1,line:size(1) do
			file_stream:write(',', line[i])
		end
		file_stream:write('\n')
	end
	file_stream:close()
end

