In the attention map section, I added a new graph with a different y-axis than the previous graph.

The previous y-axis is a sequence of a, b, c, and the second graph's y-axis is just 0, 1, 2. 

As shown in the figure, among 0,1,2, the color of 1 is brighter.

The model thinks that the character in the first position of the output sequence is 1,and the result is indeed 1. 

![33RS% I`1$W1AQ7 AA57$_O](https://user-images.githubusercontent.com/91429283/160992368-0d54425b-9903-446d-a9e7-9f0303cc283c.png)

As in this picture:

The input sequence is: aaaaa, and the output is also very accurate to give 00000222... (2 is padding). 

![I ZHQAXTE5IR9L32GUI8 3Y](https://user-images.githubusercontent.com/91429283/160992743-ce51038f-2d73-481d-a3d0-4930c9b8c0e7.png)

Through repeated tests, I think the effect of the process of ababa->0101 is very good. 

But when the y-axis is changed to a sequence consisting of a, b, and c, which is the case in the first picture,

it is very different from what we expected. 
