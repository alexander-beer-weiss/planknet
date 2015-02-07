# planknet
Planknet is a convolutional neural network for species identification of plankton.

# installation
For proper use, the clone of the planknet directory on your local machine must be placed alongside a directory containing the training data.  This directory, which must be given the name 'train', should contain some number of subdirectories, each of whose names will be used as labels for the images contained within.  (Note:  Two more sibling directories, alongside 'planknet' and 'train', will be automatically created by planknet during runtime.  These will have the names 'preprocessed' and 'misclassified'.)

# preprocessing
To load the training data, run 'th prep_data.lua' from within the planknet directory.  This will read in the jpegs and their associated labels.  It will also do all preprocessing.  It will save the preprocessed data in a sibling directory called 'preprocessed'.

# learning
To prepare the neural net for learning, run 'th -i planknet.lua -maxEpoch #' from within the planknet directory, where '#' is the number of times the neural net should train on the data.  This will automatically split the data into two sets; 80% training data and 20% cross-validation.  Once the neural net has been configured, a lua prompt will appear.  Type 'goplankton()' to actually begin the training.  Following each run through the training data, planknet will do a forward pass through the net using the cross-validation set.  Confusion matrices will be presented.

# results
After the final pass through the data, the incorrect results of the final cross-validation run will be saved to a sibling directory called 'misclassified'.  To visually inspect the misclassified images, 'cd' into the 'misclassified' directory.  Run 'itorch notebook' from the shell prompt.  This should open an iTorch window in your browser.  Create a new notebook.  From the interactive prompt, type ' dofile "misclass.lua" ' (where the quotation marks should actually be typed, but not the apostrophes).  The resulting images can be resized using your mouse.



