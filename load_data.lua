print '==> importing images'


plankton_images = {}  -- array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols)
plankton_labels = {}  -- the name of each species is read off from the directory containing the image
plankton_files = {}  -- labels for images  ( array with values of type string )
species = {}  -- an array of possible species
species_count = {}  -- number of examples of a given species



for _,dir in pairs(paths.dir(images_path)) do  -- loop through image directories (extract names of sub-directories)
	if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
		table.insert(species,dir)
		local file_count = 0
		for file in paths.files(images_path..'/'..dir) do  -- read in names of all jpeg files; make table of these names
			if string.match(file,'%.jpg$') then
				table.insert(plankton_files,file)
				table.insert(plankton_images,image.load(images_path..'/'..dir..'/'..file,1,float))
				table.insert(plankton_labels,dir)
				file_count = file_count + 1
			end	
		end
		table.insert(species_count,file_count)
	end
end


