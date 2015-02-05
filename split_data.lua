
print '==> loading preprocessed features'

preprocessed_path = '../preprocessed'


-- IMPLEMENT HDF5 READ -----------------
-- local image_files = read ../preprocessed/images.h5
-- local image_targets = read ../preprocessed/targets.h5
-- local image_paths = read ../preprocessed/paths.h5
-- local species = ../preprocessed/species.h5
----------------------------------------



-----DELETE------------DELETE------------DELETE------
-- DELETE THESE ASSIGNMENTS ONCE HDF5 READ IS WORKING
dofile 'prep_data.lua'
image_files = plankton_images
image_targets = plankton_labels
image_paths = file_names
species = species
-----DELETE------------DELETE------------DELETE------





--------------------------------------------------

print '==> seperating training from CV features'


plankton_images_train = {}  -- images of plankton ( array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols) )
plankton_images_cv = {}  -- set aside 20% for cross-validation
plankton_targets_train = {}  -- target labels for images  ( array with values of type string )
plankton_targets_cv = {}  -- set aside (same) 20% for cross-validation
plankton_paths_train = {}  -- file names, so we can visualize which ones our neural net misses
plankton_paths_cv={}

image_files = table_shuffle(image_files, image_targets, image_paths)
for i = 1,math.floor(#image_files*0.8) do -- 80% of examples used for training
	table.insert(plankton_images_train,image_files[i])
	table.insert(plankton_targets_train,image_targets[i])
	table.insert(plankton_paths_train,image_paths[i])
end
for i = 1,math.floor(#image_files*0.8)+1 do  -- 20% of examples used for cross-validation
	table.insert(plankton_images_cv,image_files[i])
	table.insert(plankton_targets_cv,image_targets[i])
	table.insert(plankton_paths_cv,image_paths[i])
end


-- store dimensions of images (note: all images are the same size due to padding)
height = plankton_images[1]:size(2)
width = plankton_images[1]:size(3)



