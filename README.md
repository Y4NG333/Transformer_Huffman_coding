# Background
We apply a Transformer, an attention based deep learning model, to learn a Huffman encoding without knowing prior distribution of the source.  
# Install
python (v3.6)    
pytorch (v1.10.2)    
huffman (v0.1.2)   
numpy (v1.19.2)   
matplotlib (v3.3.4)   
# Usage
Currently, the source code contains the files for training and testing. The *train.py* is for training the model on the training set. The *test.py* is for testing the model generated after training. it will generate the accuracy on the testing set. The training set and testing set is generated randomly in the program. Some pictures also are generated during the training and testing. The model is defined in the *model.py* and some other function such as plotting and generating the dataset is defined in the *utils.py*.    

*python train.py*   
*python test.py*  

# Example
At first, we set the length of vocab to 5 and fix a random weight. 
Then, we set the length of sequences to 10.
At last, we generate the dataset contains sequences by the vocab.

# Result
This is the output attention map and the standard attention map.  
![29990000_output](https://user-images.githubusercontent.com/91429283/182027401-8e55f87c-acb7-4661-9b48-27d23be044fc.png)
![29990000_standard](https://user-images.githubusercontent.com/91429283/182027403-599a84ef-7735-49bc-bc33-b352b062626e.png)
