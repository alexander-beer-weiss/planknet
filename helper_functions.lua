

-- takes an array of images of type torch.Tensor(1, num_rows, num_cols) -- the 1 means one channel; i.e., greyscale image
-- returns the maximum height and maximum width from the full set of images
-- optional parameter max_scan defines the maximum number of images to scan through before returning max dimensions
function maxdims(image_array, max_scan)
	local maxheight, maxwidth = 0, 0
	for i = 1,#image_array do
		if maxheight < image_array[i]:size(2) then
			maxheight = image_array[i]:size(2)
		end
		if maxwidth < image_array[i]:size(3) then
			maxwidth = image_array[i]:size(3)
		end
		if max_scan then
			max_scan = max_scan - 1
			if max_scan == 0 then
				return maxheight, maxwidth
			end
		end
	end
	return maxheight, maxwidth
end




-- IMAGE_ARRAY SHOULD ACTUALLY BE ARRAY OF IMAGES AND THIS WHOLE THING NEEDS TO BE IN A FOR LOOP
function pad(image_array, height, width, invert)
	ones_array = torch.Tensor(1,height,width):fill(1)
	for i = 1,#image_array do
		local padded_image = torch.Tensor(1,height,width):fill(1)
		local h_offset, w_offset = 0,0
		if image_array[i]:size(2) < height then
			if image_array[i]:size(3) < width then
				h_offset  = math.floor( (height - image_array[i]:size(2)) / 2 )
				w_offset  = math.floor( (width - image_array[i]:size(3)) / 2 )
				padded_image:sub(1,1,h_offset+1,h_offset+image_array[i]:size(2),w_offset+1,w_offset+image_array[i]:size(3))[1]
				   = image_array[i][1]:clone()
			else
				h_offset  = math.floor( (height - image_array[i]:size(2)) / 2 )
				w_offset  = math.floor( (image_array[i]:size(3) - width) / 2 )
				padded_image:narrow(2,h_offset+1,image_array[i]:size(2))[1]
				   = image_array[i]:narrow(3,w_offset+1,width)[1]:clone()
			end
		else
			if image_array[i]:size(3) < width then
				h_offset  = math.floor( (image_array[i]:size(2) - height) / 2 )
				w_offset  = math.floor( (width - image_array[i]:size(3)) / 2 )
				padded_image:narrow(3,w_offset+1,height)[1]
				   = image_array[i]:narrow(2,h_offset+1,image_array[i]:size(2))[1]:clone()
			else
				h_offset  = math.floor( (image_array[i]:size(2) - height) / 2 )
				w_offset  = math.floor( (image_array[i]:size(3) - width) / 2 )
				padded_image[1]
				   = image_array[i]:sub(1,1,h_offset+1,h_offset+height,w_offset+1,w_offset+width)[1]:clone()
			end
		end
		if invert then
			image_array[i] = ones_array - padded_image
		else
			image_array[i] = padded_image
		end
	end	
end





-- takes a list of arrays and reorders indices of arrays
-- assumes all tables in ... are the same length; all arrays are shuffled into same order.
-- example: takes array of plankton images and array of associated labels and shuffles such that labels still match images
function table_shuffle(...)
	local table_of_tables = {...}
	local table_length = #table_of_tables[1]
	local shuffled_indices = torch.randperm(table_length)
	shuffled_table_of_tables = {}
	
	for i = 1,#table_of_tables do
		shuffled_table_of_tables[i] = {}
		for j = 1,table_length do
			table.insert(shuffled_table_of_tables[i], table_of_tables[i][ shuffled_indices[j] ])
		end
	end
	return unpack(shuffled_table_of_tables)
end




