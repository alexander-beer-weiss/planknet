require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure

preprocessor = {}
preprocessor.__index = preprocessor
setmetatable(preprocessor, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function preprocessor.new(opt)
  local self = setmetatable({},preprocessor)
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end

function preprocessor:loadFromJPG()
  print 'Loading from JPG'
  local ones_array = torch.Tensor(1,self.height,self.width):fill(1)
  local plankton_images = {}  -- array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols)
  local plankton_labels = {}  -- the name of each species is read off from the directory containing the image
  local plankton_files = {}  -- labels for images  ( array with values of type string )
  local species = {}  -- an array of possible species
  local species_count = {}  -- number of examples of a given species
  for _,dir in pairs(paths.dir(self.trainingdir)) do  -- loop through image directories (extract names of sub-directories)
    if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
      table.insert(species,dir)
      local image_count = 0
      for file in paths.files(self.trainingdir..'/'..dir) do  -- read in names of all jpeg files; make table of these names
        if string.match(file,'%.jpg$') then
          table.insert(plankton_files,file)
          local img = image.load(self.trainingdir..'/'..dir..'/'..file,1,float)
          img = ones_array - image.scale(img,self.height,self.width)
-- Insert rotations, inversions here
          table.insert(plankton_images,img[1])
          table.insert(plankton_labels,dir)
          image_count = image_count + 1
        end
      end
    table.insert(species_count,image_count)
    end
  end
  self.images=plankton_images  
--     Names here are confusing. paths are files and targets are lables? why not just use labels and files?
  self.data={image_targets=plankton_labels, image_paths=plankton_files, species=species, species_count=species_count}
end