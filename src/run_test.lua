
require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'
require 'paths'
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-preprocesseddir', '../preprocessed', 'location of preprocessed directory')
cmd:option('-testingdir', '../test', 'location of testing directory')
cmd:option('-outputdir', '../predictions', 'location of output directory')
cmd:option('-netDatadir', '../NNsave', 'location of neural net data directory')
cmd:option('-height', 32, 'rescale height')
cmd:option('-width', 32, 'rescale width')
cmd:option('-preserveAspectRatio', false, 'if true, preserves aspect ratio of images during scaling')
cmd:text()
opt=cmd:parse(arg)


dofile 'convnet.lua'
dofile 'test_loader.lua'
test_data = test_loader(opt)
test_data:loadNet()
test_data:loadFileNames()



dofile 'test.lua'
local batch_size = 20000
local num_batches = math.ceil(#test_data.file_names / batch_size)
print('==> preparing to test ' .. #test_data.file_names .. ' images in ' .. num_batches .. ' batches of ~' .. batch_size)

local test_predictions = {}
local batch_counter = 0
for batch_counter = 1, num_batches do
	test(test_predictions, test_data.convNetObj.net, test_data:loadTestBatch(batch_counter,batch_size), batch_counter)
end

dofile(opt.preprocesseddir..'/plankton_tables.lua') -- only need species... should have been saved seperately
dofile 'csv_writer.lua'
local csv_header = {['image'] = species}
csvObject = csv_writer(opt)
csvObject:writeToCSV(test_predictions, csv_header)

