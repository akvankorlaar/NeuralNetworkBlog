# Cost function

The **cost function** represents the error between the predicted values and the actual values. We seek to minimize the cost function. Usually the cost function is implemented as the **cross-entropy** between the predicted values and the actual values. For us, because we have 2 classes, we will be using **binary cross-entropy**. 

When understanding cross-entropy, it is important to know what **entropy** is.
Entropy is a measure with which the amount of information can be quantified. It is a cornerstone of information theory and very fundamentall to much of science in general. The basic principle for entropy is that events that are likely to occur do not contain alot of information, whereas events that are unlikely to happen contain alot of information. For example:

