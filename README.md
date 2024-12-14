## 1 - Wavenet - (DeepMind)

   Aaron van Oord and other researchers from DeepMind introduced WaveNet, an innovative architecture for audio modeling, which is based on 1D dilated convolutional layers. WaveNet achieved 
   significant advancements, especially in speech synthesis tasks, such as high-quality voice synthesis, surpassing other techniques available at the time.

   1.1 - Number of parameters
    For this type of neural network architecture, the presented model has **72,617** parameters and uses stacked dilated convolution layers. The code description you provided appears to be a 
    model that applies embeddings and dense layers, as well as batch normalization (BatchNorm) and activation functions.
    
   ``` Python

    model = Sequential([
    Embedding(vocab_size, n_embd),                        # Embedding layer that transforms vocabulary indices into dense vectors of dimension n_embd
    FlattenConsecutive(2),                                  # Layer to "flatten" the dimensions, probably to facilitate passing the data to dense layers
    Linear(n_embd * 2, n_hidden, bias=False),               # Dense linear layer with n_embd * 2 inputs and n_hidden outputs, without bias
    BatchNorm1d(n_hidden),                                  # Batch normalization (BatchNorm) on the output of the previous layer
    Tanh(),                                                 # Tanh activation function
    FlattenConsecutive(2),                                  # Another layer to flatten
    Linear(n_hidden * 2, n_hidden, bias=False),             # Another dense linear layer
    BatchNorm1d(n_hidden),                                  # More batch normalization
    Tanh(),                                                 # Tanh activation function
    FlattenConsecutive(2),                                  # Another "flattening" layer
    Linear(n_hidden * 2, n_hidden, bias=False),             # Final dense linear layer
    BatchNorm1d(n_hidden),                                  # Final batch normalization
    Tanh(),                                                 # Final activation function
    Linear(n_hidden, vocab_size),                           # Output layer that generates the vocabulary prediction
])

```

      
         

