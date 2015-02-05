print '==> importing images'

images_path = '../images'



plankton_images = {}  -- array with values of type torch.DoubleTensor(num_channels, num_rows, num_cols)
plankton_labels = {}  -- the name of each species is read off from the directory containing the image
file_names = {}  -- labels for images  ( array with values of type string )
species = {}  -- an array of possible species

for _,dir in pairs(paths.dir(images_path)) do  -- loop through image directories (extract names of sub-directories)
	if not string.match(dir,'^%.') then  -- ignore any directories (or files) starting with a period
		table.insert(species,dir)
		local file_names = {}
		for file in paths.files(images_path..'/'..dir) do  -- read in names of all jpeg files; make table of these names
			if string.match(file,'%.jpg$') then
				table.insert(file_names,file)
			end	
		end
		for i = 1,#file_names do
			-- image is a torch.Tensor(num_channels, height, width)
			table.insert(plankton_images,image.load(images_path..'/'..dir..'/'..file_names[i]))  
			table.insert(plankton_labels,dir)
		end
	end
end









