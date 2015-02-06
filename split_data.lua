
print '==> seperating training from CV features'


plankton_images_train = {}  -- images of plankton ( array with values of type torch.Tensor(num_channels, num_rows, num_cols) )
plankton_images_cv = {}  -- set aside 20% for cross-validation
plankton_targets_train = {}  -- target labels for images  ( array with values of type string )
plankton_targets_cv = {}  -- set aside (same) 20% for cross-validation
plankton_paths_train = {}  -- file names, so we can visualize which ones our neural net misses
plankton_paths_cv={}

--image_files, image_targets, image_paths = table_shuffle(image_files, image_targets, image_paths)
local offset_index = 0
for i = 1,#species_count do
	local shuffle = torch.randperm(species_count[i])
	for j = 1,math.floor(species_count[i]*0.8) do -- 80% of examples used for training
		table.insert(plankton_images_train,image_files[ offset_index + shuffle[j] ])  -- need to shuffle with each species first...
		table.insert(plankton_targets_train,image_targets[ offset_index + shuffle[j] ])
		table.insert(plankton_paths_train,image_paths[ offset_index + shuffle[j] ])
	end
	for j = math.floor(species_count[i]*0.8)+1,species_count[i] do  -- 20% of examples used for cross-validation
		table.insert(plankton_images_cv,image_files[ offset_index + shuffle[j] ])
		table.insert(plankton_targets_cv,image_targets[ offset_index + shuffle[j] ])
		table.insert(plankton_paths_cv,image_paths[ offset_index + shuffle[j] ])
	end
	offset_index = offset_index + species_count[i] 
end


-- store dimensions of images (note: all images are the same size due to padding)
height = image_files[1]:size(2)
width = image_files[1]:size(3)




