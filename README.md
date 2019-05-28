# Crowd-Behaviour-Analysis #
This project aims at classifying crowd behavior as violent / non-violent.\
I used the C3D Action Classification model on a data-set and achieved around 33% accuracy.
## Dataset ##
Here is the link to the dataset that I used for training. It requires you to register before accessing the data. I had the dataset rearranged, into a single folder, for my own convenience.\
https://www.openu.ac.il/home/hassner/data/violentflows/
## Training ##
All of the training of the model has been through Google Colaboratory. The interesting part about the training has been due to the limitation of the memory with colabs. So, with GPU, colabs offers around 15GB of RAM. But the dataset when completely loaded requires more space. During the early stages, the ram would overflow and re-initialize and thus, the training would become pointless. So, I had to redo the train_c3d.py file completely to comply with the memory issues.\ 
I had to run around 500 epochs. So, I chose a batch size of 16 on the 168 video test-data. Loaded these batches and trained them 500 steps. These loading of the batches would occur at refreshed kernels, where I would specify the starting position of each of the batches (, starting from 0 and adding 16 everytime after running the train_c3d.py file).\
So the exact steps I followed are:
1. Mounted my drive in colab. All my programs and the dataset was uploaded in the drive
2. Changed directory to the specified folder.
3. Ran the train_c3d.py file. Specified the start of the batch, everytime.
4. After 500 steps are completed and the trained model is saved, I restarted the kernel and ran the steps from 1 again.
5. After the training was complete on the 168 test-data, I ran the eval_c3d.py file, that brought in the 48 test-data together and evaluated the model.

![Accuracy Curve](https://github.com/crazylazylife/Crowd-Behaviour-Analysis/blob/master/visual_logs/some_graph.JPG "Accuracy Curve")
This was what the accuracy plot of the training and testing of the model looked like.

*Note: If you wish to train the model again on the given data and have enough memory to bring in and train the training data all-together, I would suggest you to head to the C3c Github repo: https://github.com/hx173149/C3D-tensorflow and follow the steps there to train on your own custom dataset.\
My training might be faulty, owing to these constant changes in setting for training. As far as the effort given to complete the process in least specifications, I hope the training was carried out properly.*
## Future Target ##
* Improve performance of the model.
* Try out some other video classification model, like I3D, et cetra. (Need a machine with a little higher configuration) 
