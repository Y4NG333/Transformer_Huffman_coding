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

*python train.py* for training the model   
 *python test.py*  for testing the model 

# Example
As shown in the figure:

![ZLYHIK3A$ GI15U`$1O{4FU](https://user-images.githubusercontent.com/91429283/163824184-df112278-97f9-4fc7-88bb-66115d40de96.png)

![9V5`}8%FL(XOP3 )0_RA7A4](https://user-images.githubusercontent.com/91429283/163824282-b52da081-abd2-4c13-851f-f4b80688426f.png)

![Y%NAUG 2SMV0FJ9{ZQJE AO](https://user-images.githubusercontent.com/91429283/163824287-3e56f979-a6ad-4406-9a3b-447130617638.png)
