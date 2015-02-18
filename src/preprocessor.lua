require 'nn'  -- image padding (SpatialPadding.lua is a generalized version of SpatialZeroPadding.lua)
require 'image'  -- decode jpegs
require 'paths'  -- read OS directory structure



function pad(img,height,width)
	local aspect_ratio = width / height
	if img:size(3) / img:size(2) > aspect_ratio then
		local gap = img:size(3) / aspect_ratio - img:size(2)
		img = nn.SpatialPadding(math.floor(gap/2), math.ceil(gap/2), 0, 0, 1):forward(img)  -- pad with ones
	else
		local gap = img:size(2) * aspect_ratio - img:size(3)
		img = nn.SpatialPadding(0, 0, math.floor(gap/2), math.ceil(gap/2), 1):forward(img)  -- pad with ones
	end	
end



preprocessor = {}
preprocessor.__index = preprocessor
setmetatable(preprocessor, {
  __call = function (cls, ...)
    return cls.new(...)
  end,
})

function preprocessor.new(opt)
  local self = setmetatable({},preprocessor)
--   Just assuming the right parameters are here is lame and shitty
  for key, value in pairs(opt) do
    self[key] = value
  end
  return self
end


function preprocessor:loadFromJPG()
  print 'Loading from JPG'
  local plankton_images = {}  -- array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols)
  local plankton_labels = {}  -- the name of each species is read off from the directory containing the image
  local plankton_files = {}  -- labels for images  ( array with values of type string )
  local species = {}  -- an array of possible species
  local species_count = {}  -- number of examples of a given species
  local warped_multiples = {}
  for _,dir in pairs(paths.dir(self.trainingdir)) do  -- loop through image directories (extract names of sub-directories)
    if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
      table.insert(species,dir)
      local image_count = 0
      local flow = torch.Tensor(2,self.height,self.width):fill(0)  -- flow field for image.warp(src,flow)
	  local spec_multiples = 0
	  while image_count < self.minImageCount do
		spec_multiples = spec_multiples + 1
	    for file in paths.files(self.trainingdir..'/'..dir) do  -- read in names of all jpeg files; make table of these names
	      if string.match(file,'%.jpg$') then
	        table.insert(plankton_files,file)
	        local img = image.load(self.trainingdir..'/'..dir..'/'..file,1,float)
			if self.preserveAspectRatio then pad(img,self.height,self.width) end  -- add to testing code as well
	        img = image.scale(img,self.height,self.width)
	        img = img - torch.mean(img)
	        img = img/torch.std(img)
            img = image.warp(img,flow)
	        table.insert(plankton_images,img)
	        table.insert(plankton_labels,dir)
	        image_count = image_count + 1
	      end
	    end
        flow = torch.rand(2,self.height,self.width)
	  end
      table.insert(species_count, image_count / spec_multiples)
	  table.insert(warped_multiples, spec_multiples)
    end
  end
  self.images=plankton_images  
--     Names here are confusing. paths are files and targets are lables? why not just use labels and files?
  self.data={image_targets=plankton_labels, image_paths=plankton_files, species=species, species_count=species_count, warped_multiples = warped_multiples}
end

