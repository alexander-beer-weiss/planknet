require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')
dofile 'preprocessor.lua'
dofile 'hdf5io.lua'

cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-trainingdir', '../train', 'location of training directory')
cmd:option('-testingdir', '../test', 'location of testing directory')
cmd:option('-preprocesseddir', '../preprocessed', 'location of preprocessed directory')
cmd:option('-height', 32, 'rescale height')
cmd:option('-width', 32, 'rescale width')
cmd:text()
opt=cmd:parse(arg)


-- now with objects
dl = preprocessor(opt)
dl:loadFromJPG()

h5=hdf5IO(opt)
h5:writeToHDF5(dl.images, dl.data)
h5:loadFromHDF5()
