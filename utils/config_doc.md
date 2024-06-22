## How to use the config file to set hyperparameters

### dimensions: 
<Integer Value> please consider our current architecture and choose between 100 and 1000>
 Furthermore there are 3 dimensions to consider: The dimension of Hidden layer 1, hidden layer 2 and the dimension of the embedding
layer. 

### activation_function 
We use ReLU by default but you can choose from 
- "sigmoid", 
- "ReLU", 
- "leaky_relu", 
- "ELU", 
- "Hardshrink", 
- "Hardsigmoid", 
- "tanh",
- "Hardswish"


### optimizer
We Use Adam <br>
Other options include: 
- "AdamW"
- "SGD"
- "Adamax"
- "SparseAdam"
- "Adadelta"

### learning_rate
float between 0 and 1


### use_custom_linear_layer
"True" or "False"
depending on if you want to use our linear layer rather than torch's LL

---

Template
---
You can find a template version of the config file in this directory, please refrain from
changig the template as it should be only used to refer to the config syntax
