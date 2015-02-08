require 'nn'
require 'image'

convNet = {}
convNet.__index = convNet
setmetatable(convNet, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function convNet.new(opt)
  local self = setmetatable({},convNet)
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end

-- Example build
-- parameters,gradParameters = myNet:build({1,64,64,128,#species}, 2, 5)
function convNet:build(dimensions, n_pools, n_filter)
  self.net=nn.Sequential()
  local normkernel = image.gaussian1D(3)
  n_dimensions=#dimensions
  for i=1,n_dimensions-3 do
    self.net:add(nn.SpatialConvolutionMM(dimensions[i], dimensions[i+1], n_filter, n_filter))
    self.net:add(nn.Tanh())
    self.net:add(nn.SpatialLPPooling(dimensions[i+1],2,n_pools,n_pools,n_pools,n_pools))
    self.net:add(nn.SpatialSubtractiveNormalization(dimensions[i+1], normkernel))
  end
  local linearFeatDim = dimensions[n_dimensions-2] * (((self.height-n_filter+1)/n_pools - n_filter + 1 )/n_pools)  * (((width-n_filter+1)/n_pools - n_filter + 1 )/n_pools)
  print('feature dim: ' .. linearFeatDim )
  self.net:add(nn.Reshape(linearFeatDim))
  self.net:add(nn.Linear(linearFeatDim, dimensions[n_dimensions-1]))
  self.net:add(nn.Tanh())
  self.net:add(nn.Linear(dimensions[n_dimensions-1], dimensions[n_dimensions]))
  self.net:add(nn.LogSoftMax())
  
  criterion = nn.ClassNLLCriterion()
  parameters,gradParameters = self.net:getParameters()
  return parameters,gradParameters
end  
  

