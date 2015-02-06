

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



function pad(image_array, height, width, invert)
	if invert == true then
		padding = 0
	else
		padding = 1
	end


	for i = 1, #image_array do
		local padded_image = torch.Tensor(1,height,width):fill(padding)
		-- center image on padded canvas (probably not necessary for convolutional net but images display nicer in iTorch)
		local canvas_height_offset, canvas_width_offset, image_height_offset, image_width_offset = 0,0,0,0
		if image_array[i]:size(2) < height then
			canvas_height_offset = math.floor( (height - image_array[i]:size(2)) / 2 )
		else
			image_height_offset = math.floor( (image_array[i]:size(2) - height) / 2 )
		end
		if image_array[i]:size(3) < width then
			canvas_width_offset = math.floor( (width - image_array[i]:size(3)) / 2 )
		else
			image_width_offset = math.floor( (image_array[i]:size(3) - width) / 2 )
		end
		for row = 1, math.min(image_array[i]:size(2), height) do  -- oversized images are simply cropped
			for col = 1, math.min(image_array[i]:size(3), width) do
				if invert == true then
					--print(row + canvas_height_offset,col + canvas_width_offset,row + image_height_offset,col + image_width_offset)
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = 1 - image_array[i][1][row + image_height_offset][col + image_width_offset]  -- the [1] is the greyscale label index
				else
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = image_array[i][1][row + image_height_offset][col + image_width_offset]
				end
			end
		end
		image_array[i] = padded_image  -- overwrites original (unpadded) image
	end
end



--[[
-- takes an array of images of type torch.Tensor(1, num_rows, num_cols), where num_rows and num_cols vary from image to image
-- returns an array of images of type torch.Tensor(1, height, width), where height and width are same for all images
-- optional parameter invert: if true inverts matrices (i.e., components go to 1-components) yielding a sparse matrix
function pad(image_array, height, width, invert)
	if invert == true then
		padding = 0
	else
		padding = 1
	end


	for i = 1, #image_array do
		local padded_image = torch.Tensor(1,height,width):fill(padding)
		-- center image on padded canvas (probably not necessary for convolutional net but images display nicer in iTorch)
		local canvas_height_offset, canvas_width_offset, image_height_offset, image_width_offset = 0,0,0,0
		if image_array[i]:size(2) < height then
			canvas_height_offset = math.floor( (height - image_array[i]:size(2)) / 2 )
		else
			image_height_offset = math.floor( (image_array[i]:size(2) - height) / 2 )
		end
		if image_array[i]:size(3) < width then
			canvas_width_offset = math.floor( (width - image_array[i]:size(3)) / 2 )
		else
			image_width_offset = math.floor( (image_array[i]:size(3) - width) / 2 )
		end
		for row = 1, math.min(image_array[i]:size(2), height) do  -- oversized images are simply cropped
			for col = 1, math.min(image_array[i]:size(3), width) do
				if invert == true then
					--print(row + canvas_height_offset,col + canvas_width_offset,row + image_height_offset,col + image_width_offset)
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = 1 - image_array[i][1][row + image_height_offset][col + image_width_offset]  -- the [1] is the greyscale label index
				else
					padded_image[1][row + canvas_height_offset][col + canvas_width_offset]
					  = image_array[i][1][row + image_height_offset][col + image_width_offset]
				end
			end
		end
		image_array[i] = padded_image  -- overwrites original (unpadded) image
	end
end
--]]



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




