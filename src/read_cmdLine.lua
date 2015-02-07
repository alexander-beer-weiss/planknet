-- CmdLine() doesn't work in iTorch notebook.
-- Note:  Only SGD is currently fully supported.

-- parse command line arguments


print '==> parsing command line arguments'
cmd = torch.CmdLine()
cmd:text()
cmd:text('SVHN Training/Optimization')
cmd:text()
cmd:text('Options:')
cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-plot', false, 'live plot')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-maxEpoch', 100, 'maximum number of epochs during training')  -- set to -1 for unlimited epochs
cmd:text()
opt = cmd:parse(arg or {})


--[[
opt = {}
opt.save = 'results'
opt.visualize= false
opt.plot = false
opt.optimization = 'SGD'
opt.learningRate = 1e-3
opt.batchSize = 1
opt.weightDecay = 0
opt.momentum = 0
opt.t0 = 1
opt.maxIter = 2
opt.maxEpoch = 2
--]]
