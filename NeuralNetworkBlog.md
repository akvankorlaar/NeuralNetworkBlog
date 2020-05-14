# Understanding Neural Networks using real examples

A Neural Network is a computer algorithm inspired
by the human brain (hence Neural), composed of a
network of artificial neurons. This network of
aritificial neurons is able to learn patterns from
example data, and is able to use this learned knowledge to
perform some task.

Despite their consistent
rise in popularity in recent years, neural networks have been
around for quite a while: Frank Rosenblatt laid out
the fundational building blocks for the neural network in
1958. Thats over 60 years ago!

Over the years Neural Networks were improved, forgotten,
improved, and forgotten again. For years
neural networks couldn't do much, 
because well, computers couldn't do much. With the
enormous rise of computing power however, neural
networks have proven to be invaluable in many many
areas. With this technology becoming ever more
usefull and adopted with each year, it is very much worth your
time to learn how this technology works.

In this example we will attempt to learn a neural
network how to classify different types of shrubs based
on shrub height and the shrub's leave size.
We will go through the process of
preprocessing data, defining a neural network architecture,
and finally training the neural network using our preprocessed
data. Some familiarity with Python 3, Pandas and Keras
will help. 

## Input Data

The most important thing when working with Neural Networks
is not its architecture: Its the data. Why? You can
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
þLeave size (cm)       100
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
height of two different shrub species. Our tasks will be firstly to learn
the relationship between Leave size, shrub height and shrub species       
and secondly to be able
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

In this example we will be looking at a type
of Neural Network called a 'Feedforward Neural Network'.
In Feedforward Neural Network the data flow is unidirectional:
data comes in at the input, and goes out at the output.

The most fundamental block of a Neural Network is the neuron.
The neuron is a unit that takes input, does some mathematical
transformation, and produces output. For example:

equation



The weights and the bias are the learnable parameters for
the Neural Network. In other words, the Neural Network will
have to find the optimal values for these itself during training.
Most commonly at the start of the training, the weights are initialised
as small random values between 0 and 1, and the bias is initialised
at 0. 

These neurons are organised in layers. We will have 2 neurons
in the input layer, 3 neurons in the hidden layer, and 1 
neuron in the final layer to predict the class.Note that only
in the hidden layer and the final layer a transformation is computed. The input layer contains the values of the input as they are.

Every input neuron is connected to every neuron in the hidden
layer. This means that with 2 input neurons and 3 neurons in the
hidden layer we will have (2 x 3 ) 6 weights in between the two.
The hidden layer and the final layer will be connected with
3 (1 x 3) weights. THis is how our Neural Network looks like right now: 


Picture



Note that the above picture can be helpfull
to know how the architecture of a Neural Network looks, like
but that whats actually happening is this:

equation



As an example, take as input a leaf size of .. and shrub height of ...


picture



þ
There is still one important ingredient missing from our Neural Network:
The activation function.
Suppose all that happened at the layers was a multiplication with some
numbers, and an addition of some other numbers. That means that at its best
the Neural Network would be able to learn a linear function. In other words
 the output of the Neural Network would always be:

equation


This is where the activation function becomes very important. What
the activation function does is introduce some kind of nonlinearity to the
output. There are actually many different kinds of activation functions,
but the most common (and the most simple) being ReLu:


figure



Implementing the ReLu function would look like this:



code



Another kind of activiation function is the Sigmoid:




picture


An implemented version of the sigmoid looks like this:




We will add ReLu activation to our hidden layer, and
Sigmoid to our output layer. This is how our final model
will look like:


Equation




Adding ReLu and the Sigmoid activation to the example above, we now
get ....


Picture



equation 4




 
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


