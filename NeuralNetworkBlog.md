# Building a Neural Network with Python from scratch

This is a tutorial on how to write a basic Neural Network from scratch
to perform binary classification, explaining important concepts
on the way. I've been meaning to do this for   
a long time, because writing something down makes me understand   
something better, and because I love explaining stuff!

I will try to give as much practical examples as I can, 
both using math and code examples. I hope you have fun learning!

## What is a Neural Network? 

A Neural Network is an algorithm that is able to learn
a function and use this function to create new
output. More formally:





The x and y here could be literally anything. As an example
x could be the number of hours
you work out, and the y the number of calories burned. A Neural
Network could be used to approximate the relationship
(f) between these two. Whats more, once a Neural Network
has been trained, you can use it to predict calories burned
by just filling in the number of hours worked out.

The power of Neural Networks is that with the right
data and architecture they are able to learn
very complicated relationships, and they can be used
within various field. This is also
why they are very popular. Neural Networks are being used
today for all kinds of tasks such speech recognition, 
picture recognition, or even music generation.

Despite being inspired by our brain, a 
Neural Network is actually very different.
Our brain is vastly more complicated and most
of our brain is currently not very well understood, 
so an elaborate comparison will
not do any justice to our brains complexity.
That being said, there are alot of cool things
you can do with Neural Networks, and 
new applications are being created every day.

## Our example Network

In this example, using a self-built Neural Network we will
attempt to classify different kinds of shrubs based on their
leaf size and height. Mind that normally when working with
Neural Networks you will probably use a high-level API 
such as Keras or PyTorch. For the purpose of understanding
how Neural Networks work however, we will code one from scratch.

## Input Data

The most important thing when working with Neural Networks
is not the architecture, not the amount of layers, 
or the amount of weights: Its the data. Why? You can
build the largest and most complex Neural Network you want,
one rule always stays the same: Garbage in = Garbage out.
So at the input data we start! 

For this example I generated a dataset
that is located at 

First we take a look at the data to see what we have:

```python
import pandas as pd


df = pd.read_csv('shrub_dataset.csv')

df
"""
     Leave size (cm)  Shrub height (m)     Shrub species name
0           8.232398          3.064781            Hazel Shrub
1           6.374936          1.973804            Hazel Shrub
2           8.961280          3.854265            Hazel Shrub
3           8.242065          2.412739            Hazel Shrub
4           6.736104          2.559504            Hazel Shrub
...
...
97         4.047278          1.403136  Alder Buckthorn Shrub
98         5.911174          2.655614  Alder Buckthorn Shrub
99         4.131060          1.906048  Alder Buckthorn Shrub
"""

df.count()
"""
Leave size (cm)       100
Shrub height (m)      100
Shrub species name    100
"""

df['Shrub species name'].unique()

"""
array(['Hazel Shrub', 'Alder Buckthorn Shrub'],
      dtype=object)
"""
```

Fig1: Picture of the dataset


So our dataset has 100 rows, containing Leave size and shrub
height of two different shrub species. Our task will be to be able
to predict the shrub species based on its size and height using
a Neural Network.
In this example the leave size and shrub height are called the
features, and will be the Neural Networks input. The shrub species is called
the class, and will be the Neural Networks output.

Note that for all these examples we already know the shrub species. The
reason for this is that our network first needs to learn how to use
leave size and height to distinguish between different shrub species.
Data in which you already know the class you want to predict is also
called labeled data.

Before we can use this to train our network however, there
is some preprocessing we need to do. In itself the network
can only convert numbers to numbers, so we will assign the number 
'0' to the Hazel Shrub and the number '1' to
the Alder Buckthorn Shrub'.

For the input features, note that Leave size is in centimeters,
and shrub height in meters. We will convert all these input values
to 0 - 1 range. One of the reasons to do this is to prevent large features (such as
the shrub height in this case) having a disproportionate effect
on the training. Here is the full preprocessing code:

```python
import pandas as pd

df = pd.read_csv('shrub_dataset.csv')
# Extract the class column from the dataframe
# and conver the class names to numbers.
df_values = df.values
class_column = df_values[:, 2:3]
class_column[class_column == 'Hazel Shrub'] = 0
class_column[class_column == 'Alder Buckthorn Shrub'] = 1

# Drop the class column in the original df
df2 = df.drop(columns=['Shrub species name'])
# Normalise the features in the df
preprocessed_df=(df2-df2.min())/(df2.max()-df2.min())

# Insert the class column again
preprocessed_df.insert(2, 'Shrub species name', class_column)

# Write the preprocessed df to a csv
preprocessed_df.to_csv('preprocessed_shrub_dataset.csv')
```

Neural Network Architecture

The most basic unit of a Neural Network is what we call
the perceptron. 


In one of its most basic layouts a Neural Network
consists of an input layer, one or more hidden layers, and an output
layer. A neural network can have as many hidden layers as you want,
but for this example, we will start with just 1 hidden layer.

Each of these layers is composed of what we call 'neurons'. We will have 2 neurons
in the starting layer, for we have 2 features (Leave size and Shrub
height), and 1 neuron in the final layer to predict the class.
The hidden layer can have any amount of neurons. Its up to choose
how many. For this example we will choose 3 neurons.

Picture






These layers are connected with what we call 'weights'.
Every input neuron is connected to every neuron in the hidden
layer. This means that with 2 input neurons and 3 neurons in the
hidden layer we will have (2 x 3 ) 6 weights in between the two.
The hidden layer and the final layer will be connected with
3 (1 x 3) weights. The weights are the parameters whos
values have to be learned by the Neural Network. The weights
usually start at a random value before training, and
are changed during training.  

Picture2

During the forward pass, the data in the input layer is
transformed to the data in the hidden layer by multiplying
them with the weights between these layers, and finally
from the hidden layer to the output layer by again multiplying them
with the weights between the last two layers. For the purpose of demonstration,
 this is what the math behind this could look like, using the first
 row of our dataset, and random values for the weights :


 
 Fig1. Picture generated with help of http://alexlenail.me/NN-SVG/index.html




We know that the first
row in our dataset should be 1 (Hazel Shrub) from looking
at our labeled datset and not .....
This is no problem, because the whole goal of training the
Neural Network now is to calculate the difference between
this output, and the actual expected output,
and updating the weights using this information. To calculate
the difference between the actual and expected output we use
the ... .... .. With the following formula:




Fig2. Picture generated with help of http://alexlenail.me/NN-SVG/index.html



So in our case this would be


Fig3




Now that we have this number






A valid question would be 'how do I know how many hidden
layers I should choose, or how many neurons in the hidden layer'.
There is no easy way to answer this question. There are many
heuristics, but it is intirely dependent on the problem and
the data you are working with and often times involves many
attempts of trial and error. For now just remember that the
smallest network that is able to explain the variance in your
data is the best network - 


