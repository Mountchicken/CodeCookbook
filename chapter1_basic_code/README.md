# Chapter 1: Basic Code
In the world of coding, every keystroke is a decision, and like a well-tailored suit, good code never goes out of style. It's a blend of science and art, where elegance meets logic. But let’s face it, not all code is created equal. Some are like a Swiss Army knife - versatile and efficient, while others... well, they're more like a tangled ball of yarn. What separates the good from the bad? Here are five key aspects of good code:
- **Readability**:
  -  Good code should be easy to read and understand. This means using clear naming conventions, organizing code logically, and commenting where necessary to explain complex parts.
- **Simplicity and Efficiency**:
  -  Good code often follows the KISS principle ("Keep It Simple, Stupid"). It should accomplish its tasks in the simplest way possible, without unnecessary complexity.
  -  Efficient code also performs its tasks quickly and resourcefully.
- **Maintainability**:
  - Code should be easy to maintain and modify. This involves writing modular code, where different parts of the program are separated into distinct sections or functions that can be updated independently.
- **Robustness and Reliability**: Good code should handle errors gracefully and be reliable under different conditions. This involves anticipating potential issues and coding defensively against them.
- **Scalability**: The code should be able to handle growth, whether it's more data, more users, or more use cases, without significant changes to the underlying system.

In this chapter, we'll explore some of the best practices for writing good code. We'll cover everything from naming conventions to code review, and we'll also discuss some of the tools that can help you write better code. Let's get started!

### Contents
- [Chapter 1: Basic Code](#chapter-1-basic-code)
    - [Contents](#contents)
  - [1. Readability - The Art of Writing Clear Code](#1-readability---the-art-of-writing-clear-code)
    - [1.1. Docstring and Type Hinting: The Story Behind the Code](#11-docstring-and-type-hinting-the-story-behind-the-code)


## 1. Readability - The Art of Writing Clear Code
Readability in code is akin to clear handwriting in a letter. It's not just about what you write, but how you present it. A well-written piece of code should speak to its reader, guiding them through its logic as effortlessly as a well-told story. Let's delve into some of the key practices that make code readable.

### 1.1. Docstring and Type Hinting: The Story Behind the Code
A docstring, short for "documentation string," is a string literal that occurs as the first statement in a module, function, class, or method definition. Here are three most important definitions from the official Python documentation, [PEP257](https://peps.python.org/pep-0257/).

<details open>
<summary><strong>Python PEP257</strong></summary>

- **1. What Should be Documented**:
  - PEP 257 suggests that ***all public modules, functions, classes, and methods should have docstrings***. Private methods (those starting with an underscore) are considered optional for documentation but are encouraged, especially for complex code.

- **2. Docstring Format**:
  - Docstrings should be enclosed in triple double quotes (""").
  - The first line should be a short, concise summary of the object’s purpose.

- **3. Multi-line Docstrings**:
  - For longer descriptions, the summary line should be followed by a blank line, then a more elaborate description. The detailed description may include usage, arguments, return values, and raised exceptions if applicable.
  - Multi-line docstrings should end with the closing triple quotes on a line by themselves.

</details>

Here are two python templates for docstring of function and class that may give you a more concrete idea of how to write a docstring.

<details open>
<summary><strong>Template for function</strong></summary>

  ```python
  def function_name(param1, param2, ...):
      """A brief description of what the function does.

      A more detailed description of the function if necessary.

      Inputs:
          param1 (Type): Description of param1.
          param2 (Type): Description of param2.

      Returns:
          ReturnType: Description of the return value.

      Raises: (Optional)
          ExceptionType: Explanation of when and why the exception is raised.

      Notes: (Optional)
          Additional notes or examples of usage, if necessary.

      Examples: (Optional)
          >>> function_name(value1, value2)
          Expected return value
      """
      # Function implementation
      ...
  ```

</details>

<details close>
<summary><strong>Template for class</strong></summary>

  ```python
  class ClassName:
    """Brief description of the class's purpose and behavior.

    A more detailed description if necessary.

    Args:
        arg1 (Type): Description of arg1.
        arg2 (Type): Description of arg2.
        ...

    Attributes: (Optional)
        attribute1 (Type): Description of attribute1.
        attribute2 (Type): Description of attribute2.
        ...

    Methods: (Optional)
        method1: Brief description of method1.
        method2: Brief description of method2.
        ...

    Examples: (Optional)
        >>> instance = ClassName(arg1, arg2)
        >>> instance.method1()

    Notes:
        Additional information about the class, if necessary.
    """
    def __init__(self, arg1, arg2, ...):
      # Constructor implementation
      ...
  ```

</details>

Here are some more detailed examples of docstrings that you can check out:

<details close>
<summary><strong>Detailed examples for docstring</strong></summary>

```python
from typing import Union

import torch
import torch.nn as nn
from torchvision.ops.boxes import box_area

# simple functions
def box_iou(boxes1, boxes2):
    """Compute the intersection over union (IoU) between two sets of bounding boxes.

    Inputs:
        boxes1 (Tensor): Bounding boxes in format (x1, y1, x2, y2). Shape (N, 4).
        boxes2 (Tensor): Bounding boxes in format (x1, y1, x2, y2). Shape (M, 4).

    Returns:
        Union[Tensor, Tensor]: A tuple containing two tensors:
            iou (Tensor): The IoU between the two sets of bounding boxes. Shape (N, M).
            union (Tensor): The area of the union between the two sets of bounding boxes.
                Shape (N, M).
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # import ipdb; ipdb.set_trace()
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union

# simple function with dict as input
def create_conv_layer(layer_config):
    """Create a convolutional layer for a neural network based on the provided configuration.

    Inputs:
        layer_config (dict): A dictionary with the following keys:
            'in_channels' (int): The number of channels in the input.
            'out_channels' (int): The number of channels produced by the convolution.
            'kernel_size' (int or tuple): Size of the convolving kernel.
            'stride' (int or tuple, optional): Stride of the convolution. Default: 1
            'padding' (int or tuple, optional): Zero-padding added to both sides of the input.
                Default: 0

    Returns:
        nn.Module: A PyTorch convolutional layer configured according to layer_config.

    Example:
        >>> config = {'in_channels': 1, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 0}
        >>> conv_layer = create_conv_layer(config)
        >>> isinstance(conv_layer, nn.Module)
        True
    """
    return nn.Conv2d(**layer_config)

# simple class
class SimpleConvNet(nn.Module):
    """A simple convolutional neural network wrapper class extending PyTorch's nn.Module.
    This class creates a neural network with a single convolutional layer.

    Args:
        in_channels (int): The number of channels in the input.
        out_channels (int): The number of channels produced by the convolution.
        kernel_size (int or tuple): Size of the convolving kernel.

    Attributes:
        conv_layer (nn.Module): A convolutional layer as defined in the __init__ method.

    Methods:
        forward(x): Defines the forward pass of the network.

    Example:
        >>> net = SimpleConvNet(1, 16, 3)
        >>> isinstance(net, nn.Module)
        True
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(SimpleConvNet, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        """Defines the forward pass of the neural network.

        Inputs:
            x (Tensor): The input tensor to the network.

        Returns:
            Tensor: The output tensor after passing through the convolutional layer.
        """
        return self.conv_layer(x)
```

</details>
