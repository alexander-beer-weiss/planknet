require 'paths'  -- read OS directory structure
require 'hdf5'  -- read and write to hdf5 format NOTE: torch-hdf5
require 'nn'

hdf5IO = {}
hdf5IO.__index = hdf5IO
setmetatable(hdf5IO, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function hdf5IO.new(opt)
  local self = setmetatable({},hdf5IO)
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end

function hdf5IO:writeToHDF5(images,data)
  print 'Writing to HDF5'
  if not paths.dir(self.preprocesseddir) then
    print('==> creating directory '..self.preprocesseddir)
    paths.mkdir(self.preprocesseddir)
  end

  local ldat = torch.Tensor(#images,self.height,self.width)
  for i = 1,#images do
    ldat[i] = images[i]:clone()
  end
  local f = hdf5.open(self.preprocesseddir..'/preprocessed_data.h5','w')
  f:write('images',ldat)
  f:close()

-- I would have preferred to use the "official" hdf5 for Lua. this allows writing strings. 
-- This is super hacky
  local file_stream = io.open(self.preprocesseddir..'/plankton_tables.lua','w')
  for table_name,table_data in pairs(data) do
          file_stream:write(table_name,'=')
          file_stream:write('{')
          for _,i in ipairs(table_data) do
                  file_stream:write('"',i,'",')
          end
          file_stream:write('}\n')
  end
  file_stream:close()
end

function hdf5IO:loadFromHDF5()
  print 'Loading from HDF5'
  dofile(self.preprocesseddir..'/plankton_tables.lua')
  local preprocessed_file = hdf5.open(self.preprocesseddir..'/preprocessed_data.h5', 'r')
  image_files = nn.SplitTable(1):forward(preprocessed_file:read('images'):all())
  preprocessed_file:close()
end