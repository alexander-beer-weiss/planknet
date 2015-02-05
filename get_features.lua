print '==> import images'

images_path = '../images'


plankton_images_train = {}  -- images of plankton ( array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols) )
plankton_images_cv = {}  -- set aside 20% for cross-validation
plankton_labels_train = {}  -- labels for images  ( array with values of type string )
plankton_labels_cv = {}  -- set aside (same) 20% for cross-validation
plankton_files_train = {}  -- files, so we can visualize which ones our neural net misses
plankton_files_cv={}
species = {}
for _,dir in pairs(paths.dir(images_path)) do  -- loop through image directories (extract names of sub-directories)
	if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
		table.insert(species,dir)
		local image_files = {}
		for file in paths.files(images_path..'/'..dir) do  -- read in names of all jpeg files; make table of these names
			if string.match(file,'%.jpg$') then
				table.insert(image_files,file)
			end	
		end
		image_files = table_shuffle(image_files)
		local num_images = #image_files
		for i = 1, num_images do
			if i <= num_images * 0.8 then  -- 80% of examples used for training
				table.insert(plankton_images_train,image.load(images_path..'/'..dir..'/'..image_files[i]))  -- image is a torch.Tensor(num_channels, height, width)
				table.insert(plankton_labels_train,dir)  -- the directory is the species name; these are the labels
				table.insert(plankton_files_train,image_files[i])
				else  -- 20% of examples used for cross-validation
				table.insert(plankton_images_cv,image.load(images_path..'/'..dir..'/'..image_files[i]))  -- image is a torch.Tensor(num_channels, height, width)
				table.insert(plankton_labels_cv,dir)  -- the directory is the species name; these are the labels
				table.insert(plankton_files_cv,image_files[i])
			end
		end
	end
end

plankton_images_train, plankton_labels_train, plankton_files_train
   = table_shuffle(plankton_images_train,plankton_labels_train, plankton_files_train)

height, width = maxdims(plankton_images_train,100)  -- scans through images and finds the maximum dimensions of jpegs; only consider first 100 images
height, width = math.ceil(height * 1.2), math.ceil(width * 1.2)  -- we may need to accommadate larger images than what we scanned through above
                                                                       -- (images still larger than this will be cropped via pad function)
height, width = height + height % 4, width + width % 4  -- accommodate pooling (two layers of 2x2 pooling)
-- print('height: ' .. height .. '  width: ' .. width)

height,width = 128,128  -- Hack!!!  Program still crashes sometimes.  128x128 is probably a bit too small

pad(plankton_images_train, height, width, true)  -- makes all images the same dimensions (true inverts pixels so matrices become sparse)
pad(plankton_images_cv, height, width, true)