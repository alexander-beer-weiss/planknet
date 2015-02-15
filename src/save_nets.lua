
require 'paths'  -- read OS directory structure

local month =
{
	['Jan'] = '01',
	['Feb'] = '02',
	['Mar'] = '03',
	['Apr'] = '04',
	['May'] = '05',
	['Jun'] = '06',
	['Jul'] = '07',
	['Aug'] = '08',
	['Sep'] = '09',
	['Oct'] = '10',
	['Nov'] = '11',
	['Dec'] = '12'
}


local function date_stamp(date_string)
	local date = {}
	for w in string.gmatch(date_string,'%w+') do
	  table.insert(date,w)
	end
	
	return date[7] .. '_' .. month[ date[2] ] .. '_' .. date[3]
end


local function time_stamp(time_string)
	local time = {}
	for w in string.gmatch(time_string,'%w+') do
	  table.insert(time,w)
	end
	
	return time[4] .. '_' .. time[5]
end



save_nets = {}
save_nets.__index = save_nets
setmetatable(save_nets, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})



function save_nets.new(opt)
	local self = setmetatable({},save_nets)
	-- Just assuming the right parameters are here is lame and shitty
	for key, value in pairs(opt) do
		self[key] = value
	end
	return self
end



function save_nets:prepNNdirs()
	if not paths.dir(self.netDatadir) then
		print('==> creating directory ' .. self.netDatadir)
		paths.mkdir(self.netDatadir)
	elseif paths.dir(self.netDatadir .. '/newNets') then
		if paths.dir(self.netDatadir .. '/oldNets') then
			print('==> deleting directory ' .. self.netDatadir .. '/oldNets')
			paths.rmall(self.netDatadir .. '/oldNets', 'yes')
			os.execute('ls ' .. self.netDatadir)
			os.execute('ls ' .. self.netDatadir .. '/oldNets')
		end
		print('==> moving directory ' .. self.netDatadir .. '/newNets'
		       .. ' to ' .. self.netDatadir .. '/oldNets')
		os.execute('mv ' .. self.netDatadir .. '/newNets ' .. self.netDatadir .. '/oldNets')
	end
	
	print('==> creating directory ' .. self.netDatadir .. '/newNets')
	paths.mkdir(self.netDatadir .. '/newNets')
end



function save_nets:saveNN(NN_id,localNetObj)
	torch.save(self.netDatadir .. '/newNets/NN_' .. NN_id .. '.dat', localNetObj)
end



function save_nets:saveBestNet(NN_idx,date_string)
	if not paths.dir(self.netDatadir .. '/bestNets') then
		print('==> creating directory ' .. self.netDatadir .. '/bestNets')
		paths.mkdir(self.netDatadir .. '/bestNets')
	end
	local save_dir = date_stamp(date_string)
	if not paths.dir(self.netDatadir .. '/bestNets/' .. save_dir) then
		print('==> creating directory ' .. self.netDatadir .. '/bestNets/' .. save_dir)
		paths.mkdir(self.netDatadir .. '/bestNets/' .. save_dir)
	end
	
	local save_file = 'NN_' .. time_stamp(date_string) .. '.dat'
	os.execute('cp ' .. self.netDatadir .. '/newNets/NN_' .. NN_idx ..'.dat '
	                 .. self.netDatadir .. '/bestNets/' .. save_dir .. '/' .. save_file)
					 
	os.execute('ln -f -s ' .. self.netDatadir .. '/bestNets/' .. save_dir .. '/' .. save_file .. ' '
	                       .. self.netDatadir .. '/NN.dat')
end

	