This is part 2 of the blog series on the introduction on deep learning.
For part 1 check:

This part will first dive a bit (more than 1 bit actually) into entropy, and afterwards explain what the cost function of a neural network is.

# Entropy

**Entropy** is a measure with which the amount of information can be quantified. It is a cornerstone of information theory and very fundamentall to much of science in general. The basic intuition behind entropy is that events that are likely to occur do not contain alot of information, whereas events that are unlikely to happen contain alot of information.

To attempt to explain this, suppose you had a switch with 2 settings:
on or of. If you were to store the amount of information needed to know exactly the setting of the switch, how many questions would you need ?
Exactly 1. Why? You only need to ask 1 yes/no question: is the switch on?
This information can be represented as 1 bit: 0 being 'off', and 1 being 'on'. 

Now suppose we added 2 more switches. How many yes/no questions would you need now to know the exact setting of all the switches? Now you need 3 questions: Is the first switch on? Is the second switch on? Is the third switch on? So now, instead of 1 bit, we need 3 bits to represent this information.

Now the interesting thing is if we only had 1 switch, we only had 2 different settings. However, now that we have 3 switches, our switches in total have 8 different settings. So the number of bits has tripled, but the number of total possibilities has more than tripled:

```
0 0 0
0 0 1
0 1 0
0 1 1
1 0 0
1 0 1
1 1 0
1 1 1
```

So in other words, given n bits, the number of total possibilities is:

equation



This is why in information theory the entropy is equal to base-2 logarithm of the number of possibilities:

equation


When instead of the number of possibilities, the probability is used, we get the negative base-2 logarithm of the probability. 

equation


So, when using 8 possiblities for the first formula, we get:

```python
import math

math.log2(8) = 3
```

Suppose the we set the switches on a random setting, what is the chance that
it will be the following setting: 0 1 0 ? The answer is 1/8, because we have 8
settings. Using the negative log this time:

```python
import math

-math.log2(1/8) = 3
```

Still 3 bits.

# Cost function

This knowledge on entropy will help understanding the implementation of our **cost function**. The cost function in neural networks, and in machine learning in general, represents the error between the predicted values and the actual values. We seek to minimize the cost function. Usually the cost function is implemented as the **cross-entropy** between the predicted values and the actual values. For us, because we have 2 classes, we will be using **binary cross-entropy**. 

Equation


Equation 








