print '==> building convnet structure'


-- Define relevant convnet parameters

-- 2-class problem
local noutputs = #species

-- input dimensions
local nfeats = 1  -- greyscale

-- hidden units
nstates = {64,64,128}

local filtsize = 5

-- pooling will be 2 x 2 for both layers (could be done differently for different layers using a table)
local poolsize = 2



-- Build convnet structure

convnet = nn.Sequential()

-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
convnet:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
convnet:add(nn.Tanh())
--convnet:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
convnet:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
convnet:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

-- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
convnet:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
convnet:add(nn.Tanh())
convnet:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))  -- 2 means L2-norm
convnet:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

-- stage 3 : standard 2-layer neural network
local linearFeatDim = nstates[2] * (((height-filtsize+1)/poolsize - filtsize + 1 )/poolsize)  * (((width-filtsize+1)/poolsize - filtsize + 1 )/poolsize)
--print('feature dim: ' .. LinearFeatDim)
convnet:add(nn.Reshape(linearFeatDim))  -- THIS IS WHY WE CHOOSE height = width = 128.  linearFeatDim formula doesn't always work... why??
convnet:add(nn.Linear(linearFeatDim, nstates[3]))
convnet:add(nn.Tanh())
convnet:add(nn.Linear(nstates[3], noutputs))
convnet:add(nn.LogSoftMax())


-- loss function
criterion = nn.ClassNLLCriterion()
--criterion = optim.ConfusionMatrix()

-- retrieve parameters and gradients
parameters,gradParameters = convnet:getParameters()
