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
 2 - Visualizing the loss after 20,000 iterations.

   ![image](https://github.com/user-attachments/assets/bf6a98ee-ee25-46f0-a87a-41ec59ed6e8b)


3 - Evaluating the training loss and the test loss after training.

   ``` Python

      @torch.no_grad()
      def split_loss(split):
        x,y = {
            "train" : (Xtr,Ytr),
             "val" : (Xdev,Ydev),
             "test" : (Xte,yte),
        }[split]
      
        logits = model(x)
        loss = F.cross_entropy(logits,y)
        print(split,loss.item())
      
      split_loss("train")
      split_loss("val")
      split_loss("test")
```

```

   train 1.797673225402832
   val 2.061986207962036
   test 2.0607025623321533

```   

4 - Generate text after training 

  ``` Python
      
    for _ in range(20):
     out = []
     context = [0] * block_size
     while True:
       # forward pass the neural net
       logits = model(torch.tensor([context]))
       probs = F.softmax(logits, dim = 1)
       # sample from distribuition
       ix = torch.multinomial(probs,num_samples = 1).item()
       # shift the context window and  track the samples
       context = context[1:] + [ix]
       out.append(ix)
       # if we sample the special '.' token , break
       if ix == 0:
         break
   
     print(''.join(itos[i] for i in out)) # decode an print the generated word
```

```
   ruhi.
   raste.
   ruley.
   zuia.
   hiah.
   jazlyn.
   daivi.
   shono.
   saisie.
   evania.
   iat.
   rumilian.
   riah.
   ecr.
   zardyana.
   ron.
   ster.
   wynn.
   riofa.
   iste.
```












      
         

