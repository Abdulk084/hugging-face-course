# hugging-face-course


chapter 2:

Simplicity: Hardly any abstractions are made across the library. The “All in one file” is a core concept: a model’s forward pass is entirely defined in a single file, so that the code itself is understandable and hackable.

In software engineering, we would like to use different modules betweeen different apps 
shareable. This feature in ML and in transformer library is quite different and useful
and ML specific. You can find full forward pass code in a single file.

Tokenizers take care of the first and last processing steps, handling the conversion from text to numerical inputs for the neural network, and the conversion back to text when it is needed

