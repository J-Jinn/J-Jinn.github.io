# CS-396 Senior Projects I: Perpetual Work-in-Progress Status Report 4
## Author: Joseph Jinn

<br>

### Note: Refer to below for CS-396/398 Senior Project GitHub Repository

- https://github.com/J-Jinn/CS-396-398

- https://j-jinn.github.io/CS-396-398/


## Summary of Progress:
<br>
<span style = "font-family:Times New Roman;font-size:24px;font-style:normal;">
                                                                                 
A simplified version of run_generation.py with only the essential elements required to generate text prediction has been constructed.  We are currently in the process of understanding how character-level recurrent neural networks function (PyTorch Character-level RNN tutorial on the TODO list).  In general, there is a plethora of literature we should read through in order to gain a understanding of GPT2 model and its associated tokenizer.  At present, we have not delved into the source code of the GPT2 model and its associated tokenizer, but that is planned at latest Spring 2020 semester sometime and as early as Xmas and Interim 2020 Semester.

The codebase below contains a command-line based interactive program that asks the user to input a text string of some sort (anything besides a empty string).  It then asks whether the user wishes to run in "auto" or "interactive" mode.  The automated mode simply runs through text prediction generation with default settings in order to showcase the capabilities of the GPT2 model.  The interactive mode asks the user to determine the # of iterations to run the text prediction for, the "k" value to specify for retrieving the top "k" most likely words using greedy sampling, the temperature value to scale the logits by, and the next word token to choose for text prediction.  Each iteration through the loop that generates the text prediction will ask the user to choose the next word token.

We have plans to create a web-based application as a front-end for demonstration purposes.  This component of the project could possibly begin as early as the Interim Semester 2020 as we are taking Professor Pruim's course Math-W81 Data Visualization with D3, which is essentially web development for visualization data using HTML, CSS, Javascript, and the D3 library.  It may be possible to incorporate our Senior Project into the class and use the month of January to create a simple webpage that asks the user to input text and outputs the results of the generated text prediction.  It would be site similar to the Huggingface-Transformers site that demonstrates the capabilities of each of the models available in the Transformers library.

Provided Professor Kenneth Arnold is available during the duration of Xmas vaction and the Interim 2020 Semester, we plan to continue moderate amounts of work daily on this project in order to make hopefully significant progress during the month and two weeks or so before the beginning of Spring Semester 2020.  We do have plans to submit to the ACM UIST - User Interface Software and Technology Symposium provided sufficient progress has been made and there is enough time to prepare the required submission materials.

</span>

### File header for run_generation_custom.py


```python
%%capture
"""
CS-396 Senior Project I
Project: Huggingface-Transformers GPT2 Text Modeling and Prediction
Advisor: Professor Kenneth Arnold
Coordinator: Professor Keith VanderLinden
Author: Joseph Jinn

run_generation_console_program.py defines and implements a bare-bones text modeling and prediction program using the GPT2
    model and tokenizer.

#########################################################################################

Notes:

https://github.com/dunovank/jupyter-themes
(Jupyter Notebook Themes)
https://towardsdatascience.com/bringing-the-best-out-of-jupyter-notebooks-for-data-science-f0871519ca29
(useful additions for Jupyter Notebook)
https://medium.com/@rbmsingh/making-jupyter-dark-mode-great-5adaedd814db
(Jupyter dark-mode settings; my eyes are no longer bleeding...)
https://github.com/ipython-contrib/jupyter_contrib_nbextensions
(Jupyter extensions)
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
(PyTorch tutorial on character-level RNN)

Enter this in Terminal (for use with jupyter-themes):
jt -t monokai -f fira -fs 13 -nf ptsans -nfs 11 -N -kl -cursw 5 -cursc r -cellw 95% -T

#########################################################################################

Important files to reference:

modeling_gpt2.py
 - The GPT2 model source code.

tokenization_gpy2.py
 - The tokenizer class for the GPT2 model.

#########################################################################################

Reference Material to understand the Theoretical Foundation of GPT2:
https://en.wikipedia.org/wiki/Language_model
http://karpathy.github.io/2015/05/21/rnn-effectiveness/

It would also be helpful to have some concept about beam search… I’m not super-happy with what my Googling obtains but…
https://en.wikipedia.org/wiki/Beam_search
https://machinelearningmastery.com/beam-search-decoder-natural-language-processing/

Also maybe helpful but don’t get distracted:
the first 20 minutes or so of this (everything after that is details of training, skip it.)
https://www.youtube.com/watch?v=Keqep_PKrY8
https://medium.com/syncedreview/language-model-a-survey-of-the-state-of-the-art-technology-64d1a2e5a466
https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

#########################################################################################

More Notes:

- CTRL + M + L (while in command mode): Adds code cell line numbers (very useful for debugging)

- Select code fragment --> right-click --> Execute selection in Python console (Alt + Shift + E)
    - executes selected (highlighted) code without re-running entire file.

- CTRL + Q (brings up API documentation in Pycharm)

- CTRL + Space (brings up list of functions)

- Shift + Escape (close API documentation panel)
"""

#########################################################################################
#########################################################################################
```

### Import libraries.


```python
# Import required packages and libraries.
import torch  # PyTorch.
from tqdm import trange  # Instantly make your loops show a smart progress meter.
from transformers import GPT2LMHeadModel, GPT2Tokenizer

#########################################################################################
```

### Load the GPT2 model and tokenizer.


```python
# Load the GPT2-model.
model_class = GPT2LMHeadModel  # Specifies the model to use.
tokenizer_class = GPT2Tokenizer  # Specifies the tokenizer to use for the model.
tokenizer = tokenizer_class.from_pretrained('gpt2')  # Use pre-trained model.
model = model_class.from_pretrained('gpt2')  # User pre-trained model.
model.to('cpu')  # Specifies what machine to run the model on.
model.eval()  # Specifies that the model is NOT in training mode.

#########################################################################################
```




    GPT2LMHeadModel(
      (transformer): GPT2Model(
        (wte): Embedding(50257, 768)
        (wpe): Embedding(1024, 768)
        (drop): Dropout(p=0.1, inplace=False)
        (h): ModuleList(
          (0): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (1): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (2): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (3): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (4): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (5): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (6): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (7): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (8): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (9): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (10): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (11): Block(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (attn): Attention(
              (c_attn): Conv1D()
              (c_proj): Conv1D()
              (attn_dropout): Dropout(p=0.1, inplace=False)
              (resid_dropout): Dropout(p=0.1, inplace=False)
            )
            (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (mlp): MLP(
              (c_fc): Conv1D()
              (c_proj): Conv1D()
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
      (lm_head): Linear(in_features=768, out_features=50257, bias=False)
    )



### Extract top "k" most likely words to follow previous words.


```python
# noinspection DuplicatedCode,PyUnresolvedReferences
def extract_top_k_tokens(filtered_logits, k_value):
    """
    This function utilizes the torch.topk() function to choose the "k" most likely words.

    torch.topk performs a similar function to Softmax and argmax.
    Uses the words' "scores" to choose the top "k" most likely predicted words (tokens).

    - torch.topk
     - Returns the :attr:`k` largest elements of the given :attr:`input` tensor along a given dimension.

    Non-statistical and probabilistic method, so results are deterministic (always the same).

    Parameters:
        filtered_logits - entire vocabulary with assigned scores from GPT2 model.
        k_value - choose "k" top most likely words.

    Return:
        my_topk - top "k" word tokens as Tensors.
    """
    topk_debug = False

    # Return the top "k" most likely (highest score value) words in sorted order..
    my_topk = torch.topk(filtered_logits, k=k_value, dim=1, sorted=True)
    if topk_debug:
        print(f"My torch.topk object: {my_topk}\n")
        print(f"torch.topk indices: {my_topk.indices}")
        print(f"torch.topk values: {my_topk.values}\n")

    # https://stackoverflow.com/questions/34750268/extracting-the-top-k-value-indices-from-a-1-d-tensor
    # https://stackoverflow.com/questions/53903373/convert-pytorch-tensor-to-python-list

    # Indices = encoded words, Values = scores.
    if topk_debug:
        print(f"\nDecoded torch.topk indices: {[tokenizer.decode(idx) for idx in my_topk.indices.squeeze().tolist()]}")
        print(f"\nDecoded torch.topk values: {tokenizer.decode(my_topk.indices.squeeze().tolist())}\n")

        print(f"topk indices shape: {my_topk.indices.shape}")
        print(f"topk indices shape after squeeze: {my_topk.indices.squeeze().shape}")
        print(f"topk indices after squeeze: {my_topk.indices.squeeze()}\n")

        # https://stackoverflow.com/questions/43328632/pytorch-reshape-tensor-dimension

        print(f"topk indices 1st element in Tensor: {my_topk.indices[0][0]}")
        print(f"topk indices 1st element in Tensor shape: {my_topk.indices[0][0].shape}")
        print(f"topk indices 1st element in Tensor with added dimension: {my_topk.indices[0][0].unsqueeze(0)}")
        print(f"topk indices 1st element in Tensor with added dimension shape: "
              f"{my_topk.indices[0][0].unsqueeze(0).shape}\n")

    if topk_debug:
        # Ghetto looping through topk indices.
        for elements in my_topk.indices[0]:
            if topk_debug:
                print(f"topk word: {elements}")
                print(f"topk word shape: {elements.shape}")
                print(f"topk word shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")

            # Set each element as the next token for text prediction and generation.
            next_token = elements.unsqueeze(0).unsqueeze(0)
            if topk_debug:
                print(f"Next token shape: {next_token.shape}")
                print(f"Next token: {next_token}")
                print(f"Decoded next token(s): {tokenizer.decode(next_token.squeeze().tolist())}\n")

    # Returns the Tensor array of the top "k" word tokens
    return my_topk


#########################################################################################
```

### Generate and output text predictions.


```python

# noinspection DuplicatedCode,PyUnresolvedReferences
def prediction_generation(context_tokens, generated, prediction_option):
    """
    This function makes text prediction using the GPT2 model and outputs the results.

    Parameters:
       context_tokens - the encoded raw text string.
       generated - context_tokens wrapped as a PyTorch Tensor.
       prediction_option - 'auto' or 'interactive' text prediction option.
    """
    import random  # Random number generator.

    temperature = 1  # Default value.
    iterations = 20  # Default value.
    k_value = 3  # Top "k" words to choose.

    if prediction_option == "interactive":
        valid = False
        while not valid:
            print(f"\nNote: To terminate the program, enter 'exit' for any of the requested inputs.  "
                  f"Once all inputs have received a value, the program will then terminate.")

            print(f"\nNote: Temperature is a hyper-parameter of LSTMs (and neural networks generally) used to control "
                  f"the randomness of predictions by scaling the logits before applying softmax.")
            temperature = input(f"Set temperature value to (real number > 0): ")

            print(f"\nNote: This controls how many iterations to generate the top 'k' most likely word tokens based on "
                  f"the preceding token, which controls the # of word tokens the predicted text will consist of.")
            iterations = input(f"Set the number of text prediction iterations for current string to: (integer > 0): ")

            print(f"\nNote: This controls the # of tokens returned by the torch.topk() greedy sampling function.")
            k_value = input(f"Enter the 'k' value for top 'k' most likely word token generation (integer > 0): ")

            if temperature == "exit" or iterations == "exit" or k_value == "exit":
                print(f"Terminating program...")
                quit(0)

            try:
                if float(temperature) > 0.0 and int(iterations) > 0 and int(k_value) > 0:
                    valid = True
                else:
                    print(f"Invalid value(s) detected! Please choose valid value(s)!\n")
            except TypeError:
                continue

    generated_array = []  # List of "generated" PyTorch Tensor containing encoded word tokens.
    token_score_array = []  # List of "scores" for each token in the current iteration of topk greedy sampling.
    alternative_route = []  # List containing the alternative choices we could have made.

    logits_debug = False
    topk_debug = False
    output_debug = False
    alternative_debug = False

    # Create list of PyTorch Tensors containing encoded original raw text string.
    # Create a list of word token score values initially set to 1.0.
    for i in range(0, int(k_value)):
        generated_array.append(generated)
        token_score_array.append(1.)

    chosen_generated = generated_array[0]  # For initial iteration.

    # Setup for displaying alternative routes.
    for element in range(0, int(k_value)):
        alternative_route.append(tokenizer.decode(context_tokens))
    if alternative_debug:
        print(f"")
        for element in alternative_route:
            print(f"Contents of alternative_route nested lists: {element}")

    ############################################################################################

    with torch.no_grad():  # This specifies not to use stochastic gradient descent!
        for _ in trange(int(iterations)):

            ############################################################################################

            # Note: Feeding the results back into the model is the beginnings of a beam search algorithm.
            # Currently, randomly chooses one of the "generated" Tensors to feed back in.
            if logits_debug:
                print(f"Original generated shape: {generated}")
                print(f"Generated array element 0 shape: {generated_array[0]}")
                print(f"token_score_array element 0 shape: {token_score_array[0]}\n")

            if prediction_option == "auto":
                chosen_generated = generated_array[random.randint(0, int(k_value) - 1)]

            # Call to GPT2 model generates a Tensor object containing "scores" for the entire vocabulary.
            outputs = model(input_ids=chosen_generated)
            if logits_debug:
                print(f"Outputs shape: {list(outputs)[0].shape}\n")
                print(f"Outputs: {list(outputs)[0]}\n")  # Outputs is a tensor containing a lot of stuff...

            next_token_logits = outputs[0][:, -1, :] / (float(temperature) if float(temperature) > 0 else 1.)
            if logits_debug:
                print(f"Next token logits shape: {next_token_logits.shape}\n")
                print(f"Next token logits: {next_token_logits}\n")

            filtered_logits = next_token_logits  # Set to default name from run_generation.py

            ############################################################################################

            # Call function to extract the top "k" word tokens based on their scores.
            my_topk = extract_top_k_tokens(filtered_logits, int(k_value))

            if prediction_option == "auto":
                # Ghetto looping through topk indices.
                counter = 0
                for elements in my_topk.indices[0]:
                    if topk_debug:
                        print(f"topk word: {elements}")
                        print(f"topk word shape: {elements.shape}")
                        print(f"topk word shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")

                    # Set each element as the next token for text prediction and generation.
                    next_token = elements.unsqueeze(0).unsqueeze(0)
                    if topk_debug:
                        print(f"Next token shape: {next_token.shape}")
                        print(f"Next token: {next_token}")
                        print(f"Decoded next token(s): {tokenizer.decode(next_token.squeeze().tolist())}\n")

                    # Concatenate the chosen token (predicted word) to the end of the tokenized (encoded) string.
                    # Then, add to the array of "generated" PyTorch tensors by modifying the original generated.
                    generated_array[counter] = (torch.cat((chosen_generated, next_token), dim=1))
                    if topk_debug:
                        print(f"Generated shape: {chosen_generated.shape}")
                        print(f"Generated: {chosen_generated}")
                        print(f"Decoded 'generated' tokens: {tokenizer.decode(chosen_generated.squeeze().tolist())}\n")

                    counter += 1

                ############################################################################################

                # Output the text prediction results.
                print(f"\n###############################################################################")
                print(f"Note: The '#' at the beginning and end delimit the start and end of the text.")
                print(f"Original (excluding text prediction) raw text string: {tokenizer.decode(context_tokens)}\n")
                counter = 0
                for gen in generated_array:
                    out = gen
                    if output_debug:
                        print(f"Contents of 'out': {out}")

                    # This line removes the original text but keeps appending the generated words one-by-one
                    # (based on iteration length).
                    out = out[:, len(context_tokens):].tolist()
                    if output_debug:
                        print(f"Contents of 'out' after .tolist(): {out}\n")
                        print(f"Length of context tokens:{len(context_tokens)}\n")

                    # Outputs the result of the text modeling and prediction.
                    for o in out:
                        # Decode - convert from token ID's back into English words.
                        text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                        #     text = text[: text.find(args.stop_token) if args.stop_token else None]
                        print(f"Prediction {counter} of {int(k_value - 1)} for this iteration based on previous "
                              f"iterations' randomly selected tokens (using RNG).")
                        print(f"Predicted (excluding original raw input text) text string: #{text}#")
                    counter += 1
                print(f"###############################################################################\n")

            ############################################################################################

            if prediction_option == "interactive":
                chosen = False
                while not chosen:
                    print(f"\nEnter 'exit program' or 'Exit Program' to terminate the program.")
                    print(f"The top k={k_value} tokens are:")
                    print(f"Note: The '#' are there to delimit the start and end of the token since tokens "
                          f"can include '\\n' and other invisible characters.")
                    print(f"Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.\n")
                    counter = 0
                    for elements in my_topk.indices[0]:
                        print(f"Token {counter}: #{tokenizer.decode(elements.unsqueeze(0).tolist())}#")
                        # print(f"{type(tokenizer.decode(elements.unsqueeze(0).tolist()))}")
                        alternative_route[counter] = alternative_route[counter] + tokenizer.decode(
                            elements.unsqueeze(0).tolist())
                        counter += 1

                    choose_token = input(f"\nChoose a token to use for the next iteration of text prediction:")

                    if choose_token == "exit program" or choose_token == "Exit Program":
                        print(f"Terminating program...")
                        quit(0)

                    for elements in my_topk.indices[0]:
                        if choose_token == str(tokenizer.decode(elements.unsqueeze(0).tolist())):
                            next_token = elements.unsqueeze(0).unsqueeze(0)
                            chosen_generated = (torch.cat((chosen_generated, next_token), dim=1))
                            chosen = True
                            break

                ############################################################################################

                # Output the text prediction results.
                print(f"\n###############################################################################")
                print(f"Original (excluding text prediction) raw text string: {tokenizer.decode(context_tokens)}\n")

                print(f"All routes that user could have made in choosing a token from the current iteration:")
                counter = 0
                for i in range(0, int(k_value)):
                    print(f"Route {counter}: {alternative_route[counter]}")
                    counter += 1

                out = chosen_generated
                if output_debug:
                    print(f"Contents of 'out': {out}")

                # This line removes the original text but keeps appending the generated words one-by-one (based on
                # iteration length).
                out = out[:, len(context_tokens):].tolist()
                if output_debug:
                    print(f"Contents of 'out' after .tolist(): {out}\n")
                    print(f"Length of context tokens:{len(context_tokens)}\n")

                # Outputs the result of the text modeling and prediction.
                for o in out:
                    # Decode - convert from token ID's back into English words.
                    text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(args.stop_token) if args.stop_token else None]
                    print(f"Note: The '#' at the beginning and end delimit the start and end of the text.")
                    print(f"Predicted (excluding original raw input text) text string: #{text}#")
                    print(f"###############################################################################\n")

            # Updated setup for displaying alternative routes based on each iteration's chosen token.
            # This essentially means we display alternative routes based on the previously chosen token(s).
            counter = 0
            for element in range(0, int(k_value)):
                alternative_route[counter] = (tokenizer.decode(context_tokens) + text)
                counter += 1

            ############################################################################################

            # Store the scores for each token.
            counter = 0
            for elements in my_topk.values[0]:
                token_score_array[counter] = elements.unsqueeze(0).unsqueeze(0)
                if topk_debug:
                    print(f"topk word score: {elements}")
                    print(f"topk word score shape: {elements.shape}")
                    print(f"topk word score shape after un-squeezing: {elements.unsqueeze(0).unsqueeze(0).shape}")
                counter += 1

#########################################################################################
```

### The usual main function...


```python
def main():
    """
    Main encodes the raw text string, wraps in PyTorch Tensor, and calls prediction_generation().
    Executes forever until user enters "exit" or "Exit".

    Parameters: None
    Return: None
    """
    main_debug = False
    context_debug = False
    num_samples = 1  # Default value.
    user_option = "auto"

    print(f"Welcome to the GPT2 bare-bones run_generation.py test.")
    print(f"Note: Enter 'exit' or 'Exit' to quit the program.")

    print(f"Please choose between automated text prediction or interactive text prediction:\n"
          f"Automated chooses default hard-coded settings and proceeds on its own.\n"
          f"Interactive allows the user to choose the next token used in text prediction "
          f"and adjust some other settings.\n")

    repeat_query = True
    while repeat_query:
        user_option = input(f"Type 'auto' or 'interactive'.")

        if user_option == "exit" or user_option == "Exit":
            return
        elif user_option != "auto" and user_option != "interactive":
            repeat_query = True
            print(f"Unrecognized option - type 'auto' or 'interactive'!")
        else:
            repeat_query = False

    ############################################################################################

    while True:
        raw_text = ""
        while len(raw_text) == 0:
            raw_text = input("Enter a string: ")
            if len(raw_text) == 0:
                print(f"Please enter something that is NOT a empty string!")

        # Quit the program.
        if raw_text == "exit" or raw_text == "Exit":
            print(f"Terminating program execution.")
            break

        # Encode raw text.
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        if main_debug:
            print(f"Raw text: {raw_text}\n")
            print(f"Context tokens: {context_tokens}\n")

        context = context_tokens  # Set to name as in run_generation.py

        # Convert to a PyTorch Tensor object (numpy array).
        context = torch.tensor(context, dtype=torch.long, device='cpu')
        if context_debug:
            print(f"Context shape: {context.shape}")
            print(f"Context converted to PyTorch Tensor object: {context}\n")

        # Un-squeeze adds a dimension to the Tensor array.
        # Repeat adds x-dimensions and repeats the Tensor elements y-times.
        context = context.unsqueeze(0).repeat(num_samples, 1)
        if context_debug:
            print(f"Context shape after 'un-squeeze': {context.shape}")
            print(f"Context after 'un-squeeze': {context}\n")

        generated = context  # Set to name as in run_generation.py

        # Generate and output text prediction results.
        prediction_generation(context_tokens, generated, user_option)
        print(f"Iterations for current string has ended.  Will request user enter new string.\n")


#########################################################################################
```

### "Auto" Mode program execution. (old version)


```python
#########################################################################################

# Execute the program.
# In Pycharm, select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
if __name__ == '__main__':
    main()

############################################################################################
```

    Welcome to the GPT2 bare-bones run_generation.py test.
    Note: Enter 'exit' or 'Exit' to quit the program.
    Please choose between automated text prediction or interactive text prediction:
    Automated chooses default hard-coded settings and proceeds on its own.
    Interactive allows the user to choose the next token used in text prediction and adjust some other settings.
    
    Type 'auto' or 'interactive'.auto
    Enter a string: I am thinking of
    

      0%|          | 0/20 [00:00<?, ?it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: ##
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: ##
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: ##
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going#
    ###############################################################################
    

      5%|▌         | 1/20 [00:00<00:03,  5.47it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going with#
    ###############################################################################
    

     10%|█         | 2/20 [00:00<00:03,  5.24it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going with#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going with#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back to#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back for#
    ###############################################################################
    

     15%|█▌        | 3/20 [00:00<00:03,  5.64it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back for#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and looking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back for#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and looking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and re#
    ###############################################################################
    

     20%|██        | 4/20 [00:00<00:02,  6.01it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and looking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and re#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and re#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing it#
    ###############################################################################
    

     25%|██▌       | 5/20 [00:00<00:02,  6.31it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing it#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a book#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing it#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a book#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a lot#
    ###############################################################################
    

     30%|███       | 6/20 [00:00<00:02,  6.38it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little bit#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a book#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a lot#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little bit#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a lot#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little bit#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little research#
    ###############################################################################
    

     35%|███▌      | 7/20 [00:01<00:02,  6.44it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more research#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little research#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more research#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little research#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more research#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work#
    ###############################################################################
    

     40%|████      | 8/20 [00:01<00:01,  6.23it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work with#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work with#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work.#
    ###############################################################################
    

     45%|████▌     | 9/20 [00:01<00:01,  6.30it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work with#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work.#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on my#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work.#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on my#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on this#
    ###############################################################################
    

     50%|█████     | 10/20 [00:01<00:01,  6.29it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the project#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on my#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on this#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the project#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the game#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on this#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the project#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the game#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other#
    ###############################################################################
    

     55%|█████▌    | 11/20 [00:01<00:01,  6.29it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other side#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the game#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other side#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other projects#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other side#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other projects#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff#
    ###############################################################################
    

     60%|██████    | 12/20 [00:01<00:01,  6.17it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other projects#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff,#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff,#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff I#
    ###############################################################################
    

     65%|██████▌   | 13/20 [00:02<00:01,  6.16it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff,#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff I#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.
    #
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff I#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.
    #
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. It#
    ###############################################################################
    

     70%|███████   | 14/20 [00:02<00:00,  6.08it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I'm#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff.
    #
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. It#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I'm#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I think#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. It#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I'm#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I think#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am#
    ###############################################################################
    

     75%|███████▌  | 15/20 [00:02<00:00,  5.97it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am not#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I think#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am not#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am not#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am going#
    ###############################################################################
    

     80%|████████  | 16/20 [00:02<00:00,  5.97it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am going#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking about#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am going#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking about#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking more#
    ###############################################################################
    

     85%|████████▌ | 17/20 [00:02<00:00,  5.94it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking about#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking more#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of going#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking more#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of going#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of getting#
    ###############################################################################
    

     90%|█████████ | 18/20 [00:02<00:00,  5.87it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of going#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of getting#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of getting#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing a#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing more#
    ###############################################################################
    

     95%|█████████▌| 19/20 [00:03<00:00,  5.84it/s]

    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing more#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing more#
    ###############################################################################
    Original raw text string: I am thinking of
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some more#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some of#
    ###############################################################################
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # going back and doing a little more work on the other stuff. I am thinking of doing some other#
    ###############################################################################
    

    100%|██████████| 20/20 [00:03<00:00,  6.05it/s]
    

    Iterations for current string has ended.  Will request user enter new string.
    
    Enter a string: exit
    Terminating program execution.
    

### "Interactive" Mode program execution. (old version)


```python
#########################################################################################

# Execute the program.
# In Pycharm, select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
if __name__ == '__main__':
    main()

############################################################################################
```

    Welcome to the GPT2 bare-bones run_generation.py test.
    Note: Enter 'exit' or 'Exit' to quit the program.
    Please choose between automated text prediction or interactive text prediction:
    Automated chooses default hard-coded settings and proceeds on its own.
    Interactive allows the user to choose the next token used in text prediction and adjust some other settings.
    
    Type 'auto' or 'interactive'.interactive
    Enter a string: Today is
    
    Note: To terminate the program, enter 'exit' for any of the requested inputs.  Once all inputs have received a value, the program will then terminate.
    
    Note: Temperature is a hyper-parameter of LSTMs (and neural networks generally) used to control the randomness of predictions by scaling the logits before applying softmax.
    Set temperature value to (real number > 0): 1
    
    Note: This controls how many iterations to generate the top 'k' most likely word tokens based on the preceding token, which controls the # of word tokens the predicted text will consist of.
    Set the number of text prediction iterations for current string to: (integer > 0): 20
    
    Note: This controls the # of tokens returned by the torch.topk() greedy sampling function.
    Enter the 'k' value for top 'k' most likely word token generation (integer > 0): 5
    

      0%|          | 0/20 [00:00<?, ?it/s]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # the#
    Token 0: # a#
    Token 0: # an#
    Token 0: # not#
    Token 0: # when#
    
    Choose a token to use for the next iteration of text prediction: the
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the#
    ###############################################################################
    

      5%|▌         | 1/20 [00:30<09:44, 30.76s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # day#
    Token 0: # time#
    Token 0: # first#
    Token 0: # end#
    Token 0: # moment#
    
    Choose a token to use for the next iteration of text prediction: first
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first#
    ###############################################################################
    

     10%|█         | 2/20 [00:48<08:02, 26.79s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # time#
    Token 0: # day#
    Token 0: # year#
    Token 0: # of#
    Token 0: # week#
    
    Choose a token to use for the next iteration of text prediction: time
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time#
    ###############################################################################
    

     15%|█▌        | 3/20 [00:59<06:17, 22.21s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # that#
    Token 0: # in#
    Token 0: # I#
    Token 0: # we#
    Token 0: # a#
    
    Choose a token to use for the next iteration of text prediction: that
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that#
    ###############################################################################
    

     20%|██        | 4/20 [01:05<04:36, 17.26s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # the#
    Token 0: # I#
    Token 0: # a#
    Token 0: # we#
    Token 0: # an#
    
    Choose a token to use for the next iteration of text prediction: I
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I#
    ###############################################################################
    

     25%|██▌       | 5/20 [01:11<03:28, 13.87s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: #'ve#
    Token 0: # have#
    Token 0: #'m#
    Token 0: # am#
    Token 0: # can#
    
    Choose a token to use for the next iteration of text prediction: have
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have#
    ###############################################################################
    

     30%|███       | 6/20 [01:16<02:38, 11.30s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # ever#
    Token 0: # been#
    Token 0: # seen#
    Token 0: # had#
    Token 0: # heard#
    
    Choose a token to use for the next iteration of text prediction: ever
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever#
    ###############################################################################
    

     35%|███▌      | 7/20 [01:25<02:15, 10.43s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # been#
    Token 0: # seen#
    Token 0: # had#
    Token 0: # heard#
    Token 0: # met#
    
    Choose a token to use for the next iteration of text prediction: seen
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen#
    ###############################################################################
    

     40%|████      | 8/20 [01:29<01:42,  8.54s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # a#
    Token 0: # the#
    Token 0: # an#
    Token 0: # such#
    Token 0: # this#
    
    Choose a token to use for the next iteration of text prediction: such
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such#
    ###############################################################################
    

     45%|████▌     | 9/20 [01:33<01:18,  7.12s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # a#
    Token 0: # an#
    Token 0: # great#
    Token 0: # amazing#
    Token 0: # strong#
    
    Choose a token to use for the next iteration of text prediction: a
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a#
    ###############################################################################
    

     50%|█████     | 10/20 [01:40<01:11,  7.16s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # large#
    Token 0: # thing#
    Token 0: # beautiful#
    Token 0: # great#
    Token 0: # huge#
    
    Choose a token to use for the next iteration of text prediction: beautiful
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful#
    ###############################################################################
    

     55%|█████▌    | 11/20 [01:45<00:58,  6.53s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: #,#
    Token 0: # and#
    Token 0: # woman#
    Token 0: # piece#
    Token 0: # image#
    
    Choose a token to use for the next iteration of text prediction: woman
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman#
    ###############################################################################
    

     60%|██████    | 12/20 [01:54<00:57,  7.17s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # in#
    Token 0: #.#
    Token 0: #,#
    Token 0: # and#
    Token 0: # with#
    
    Choose a token to use for the next iteration of text prediction: with
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with#
    ###############################################################################
    

     65%|██████▌   | 13/20 [01:59<00:47,  6.77s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # a#
    Token 0: # such#
    Token 0: # her#
    Token 0: # the#
    Token 0: # so#
    
    Choose a token to use for the next iteration of text prediction: her
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her#
    ###############################################################################
    

     70%|███████   | 14/20 [02:11<00:49,  8.32s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # hair#
    Token 0: # beautiful#
    Token 0: # own#
    Token 0: # face#
    Token 0: # head#
    
    Choose a token to use for the next iteration of text prediction: face
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face#
    ###############################################################################
    

     75%|███████▌  | 15/20 [02:19<00:41,  8.21s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # covered#
    Token 0: # and#
    Token 0: # in#
    Token 0: # painted#
    Token 0: # so#
    
    Choose a token to use for the next iteration of text prediction: covered
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face covered#
    ###############################################################################
    

     80%|████████  | 16/20 [02:28<00:33,  8.32s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # in#
    Token 0: # with#
    Token 0: #.#
    Token 0: # by#
    Token 0: #,#
    
    Choose a token to use for the next iteration of text prediction: in
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face covered in#
    ###############################################################################
    

     85%|████████▌ | 17/20 [02:34<00:22,  7.62s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # makeup#
    Token 0: # a#
    Token 0: # lipstick#
    Token 0: # blood#
    Token 0: # her#
    
    Choose a token to use for the next iteration of text prediction: blood
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face covered in blood#
    ###############################################################################
    

     90%|█████████ | 18/20 [02:42<00:15,  7.80s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: #.#
    Token 0: # and#
    Token 0: #,#
    Token 0: #."#
    Token 0: #,"#
    
    Choose a token to use for the next iteration of text prediction: and
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face covered in blood and#
    ###############################################################################
    

     95%|█████████▌| 19/20 [02:48<00:07,  7.37s/it]

    
    Enter 'exit' or 'Exit' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # her#
    Token 0: # with#
    Token 0: # blood#
    Token 0: # eyes#
    Token 0: # a#
    
    Choose a token to use for the next iteration of text prediction: her
    Original raw text string: Today is
    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted text: # the first time that I have ever seen such a beautiful woman with her face covered in blood and her#
    ###############################################################################
    

    100%|██████████| 20/20 [03:05<00:00,  9.28s/it]
    

    Iterations for current string has ended.  Will request user enter new string.
    
    Enter a string: exit
    Terminating program execution.
    

### Updated "auto" mode program execution.


```python
# Execute the program.
# Select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
if __name__ == '__main__':
    main()

############################################################################################
```

    Welcome to the GPT2 bare-bones run_generation.py test.
    Note: Enter 'exit' or 'Exit' to quit the program.
    Please choose between automated text prediction or interactive text prediction:
    Automated chooses default hard-coded settings and proceeds on its own.
    Interactive allows the user to choose the next token used in text prediction and adjust some other settings.
    
    Type 'auto' or 'interactive'.auto
    Enter a string: print("Hello
    

      0%|          | 0/20 [00:00<?, ?it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: #,#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # world#
    ###############################################################################
    
    

      5%|▌         | 1/20 [00:00<00:02,  7.69it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World!"#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World!#
    ###############################################################################
    
    

     10%|█         | 2/20 [00:00<00:02,  7.32it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    #
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World"));#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World").#
    ###############################################################################
    
    

     15%|█▌        | 3/20 [00:00<00:02,  6.21it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    
    #
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    "#
    ###############################################################################
    
    

     20%|██        | 4/20 [00:00<00:02,  5.87it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .set#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .append#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add#
    ###############################################################################
    
    

     25%|██▌       | 5/20 [00:00<00:02,  5.31it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add_#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .addEvent#
    ###############################################################################
    
    

     30%|███       | 6/20 [00:01<00:02,  5.65it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add( "#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(function#
    ###############################################################################
    
    

     35%|███▌      | 7/20 [00:01<00:02,  5.70it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new String#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Text#
    ###############################################################################
    
    

     40%|████      | 8/20 [00:01<00:01,  6.00it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date()#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date());#
    ###############################################################################
    
    

     45%|████▌     | 9/20 [00:01<00:01,  6.35it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2015#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2016#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011#
    ###############################################################################
    
    

     50%|█████     | 10/20 [00:01<00:01,  6.65it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011-#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011.#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/#
    ###############################################################################
    
    

     55%|█████▌    | 11/20 [00:01<00:01,  7.16it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/10#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/12#
    ###############################################################################
    
    

     60%|██████    | 12/20 [00:01<00:01,  7.49it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01/#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01-#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01,#
    ###############################################################################
    
    

     65%|██████▌   | 13/20 [00:02<00:00,  7.65it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 1#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 2#
    ###############################################################################
    
    

     70%|███████   | 14/20 [00:02<00:00,  7.88it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11:#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11am#
    ###############################################################################
    
    

     75%|███████▌  | 15/20 [00:02<00:00,  7.98it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.01#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.30#
    ###############################################################################
    
    

     80%|████████  | 16/20 [00:02<00:00,  7.91it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11.#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11:#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")#
    ###############################################################################
    
    

     85%|████████▌ | 17/20 [00:02<00:00,  8.01it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11"));#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11") +#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")).#
    ###############################################################################
    
    

     90%|█████████ | 18/20 [00:02<00:00,  7.99it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11"));
    #
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")); //#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")); }#
    ###############################################################################
    
    

     95%|█████████▌| 19/20 [00:02<00:00,  7.91it/s]

    
    ###############################################################################
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Original (excluding text prediction) raw text string: print("Hello
    
    Prediction 0 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")); // add#
    Prediction 1 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")); // Add#
    Prediction 2 of 2 for this iteration based on previous iterations' randomly selected tokens (using RNG).
    Predicted (excluding original raw input text) text string: # World")
    .add(new Date("2011/01, 11.11")); // Create#
    ###############################################################################
    
    

    100%|██████████| 20/20 [00:02<00:00,  6.91it/s]
    

    Iterations for current string has ended.  Will request user enter new string.
    
    Enter a string: exit
    Terminating program execution.
    

### Updated "interactive" mode program execution.


```python
# Execute the program.
# Select below and Run with "Alt + Shift + E" to avoid re-running entire fire and re-loading model every-time.
if __name__ == '__main__':
    main()

############################################################################################
```

    Welcome to the GPT2 bare-bones run_generation.py test.
    Note: Enter 'exit' or 'Exit' to quit the program.
    Please choose between automated text prediction or interactive text prediction:
    Automated chooses default hard-coded settings and proceeds on its own.
    Interactive allows the user to choose the next token used in text prediction and adjust some other settings.
    
    Type 'auto' or 'interactive'.interactive
    Enter a string: Emacs is
    
    Note: To terminate the program, enter 'exit' for any of the requested inputs.  Once all inputs have received a value, the program will then terminate.
    
    Note: Temperature is a hyper-parameter of LSTMs (and neural networks generally) used to control the randomness of predictions by scaling the logits before applying softmax.
    Set temperature value to (real number > 0): 1
    
    Note: This controls how many iterations to generate the top 'k' most likely word tokens based on the preceding token, which controls the # of word tokens the predicted text will consist of.
    Set the number of text prediction iterations for current string to: (integer > 0): 10
    
    Note: This controls the # of tokens returned by the torch.topk() greedy sampling function.
    Enter the 'k' value for top 'k' most likely word token generation (integer > 0): 5
    

      0%|          | 0/10 [00:00<?, ?it/s]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # a#
    Token 1: # the#
    Token 2: # an#
    Token 3: # not#
    Token 4: # now#
    
    Choose a token to use for the next iteration of text prediction: the
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is a
    Route 1: Emacs is the
    Route 2: Emacs is an
    Route 3: Emacs is not
    Route 4: Emacs is now
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the#
    ###############################################################################
    
    

     10%|█         | 1/10 [00:03<00:33,  3.74s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # most#
    Token 1: # only#
    Token 2: # first#
    Token 3: # best#
    Token 4: # default#
    
    Choose a token to use for the next iteration of text prediction: most
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most
    Route 1: Emacs is the only
    Route 2: Emacs is the first
    Route 3: Emacs is the best
    Route 4: Emacs is the default
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most#
    ###############################################################################
    
    

     20%|██        | 2/10 [00:09<00:35,  4.42s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # popular#
    Token 1: # powerful#
    Token 2: # common#
    Token 3: # widely#
    Token 4: # important#
    
    Choose a token to use for the next iteration of text prediction: common
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most popular
    Route 1: Emacs is the most powerful
    Route 2: Emacs is the most common
    Route 3: Emacs is the most widely
    Route 4: Emacs is the most important
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common#
    ###############################################################################
    
    

     30%|███       | 3/10 [00:23<00:50,  7.14s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # language#
    Token 1: # programming#
    Token 2: # way#
    Token 3: # type#
    Token 4: # Lisp#
    
    Choose a token to use for the next iteration of text prediction: way
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common language
    Route 1: Emacs is the most common programming
    Route 2: Emacs is the most common way
    Route 3: Emacs is the most common type
    Route 4: Emacs is the most common Lisp
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way#
    ###############################################################################
    
    

     40%|████      | 4/10 [00:39<00:58,  9.79s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # to#
    Token 1: # of#
    Token 2: # for#
    Token 3: # you#
    Token 4: # that#
    
    Choose a token to use for the next iteration of text prediction: to
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to
    Route 1: Emacs is the most common way of
    Route 2: Emacs is the most common way for
    Route 3: Emacs is the most common way you
    Route 4: Emacs is the most common way that
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to#
    ###############################################################################
    
    

     50%|█████     | 5/10 [00:45<00:43,  8.65s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # write#
    Token 1: # create#
    Token 2: # use#
    Token 3: # get#
    Token 4: # access#
    
    Choose a token to use for the next iteration of text prediction: write
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to write
    Route 1: Emacs is the most common way to create
    Route 2: Emacs is the most common way to use
    Route 3: Emacs is the most common way to get
    Route 4: Emacs is the most common way to access
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to write#
    ###############################################################################
    
    

     60%|██████    | 6/10 [00:52<00:32,  8.16s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # a#
    Token 1: # code#
    Token 2: # Emacs#
    Token 3: # programs#
    Token 4: # Lisp#
    
    Choose a token to use for the next iteration of text prediction: code
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to write a
    Route 1: Emacs is the most common way to write code
    Route 2: Emacs is the most common way to write Emacs
    Route 3: Emacs is the most common way to write programs
    Route 4: Emacs is the most common way to write Lisp
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to write code#
    ###############################################################################
    
    

     70%|███████   | 7/10 [01:06<00:29,  9.91s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: #.#
    Token 1: # in#
    Token 2: #,#
    Token 3: # with#
    Token 4: # for#
    
    Choose a token to use for the next iteration of text prediction: with
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to write code.
    Route 1: Emacs is the most common way to write code in
    Route 2: Emacs is the most common way to write code,
    Route 3: Emacs is the most common way to write code with
    Route 4: Emacs is the most common way to write code for
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to write code with#
    ###############################################################################
    
    

     80%|████████  | 8/10 [01:11<00:17,  8.59s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: # Emacs#
    Token 1: # the#
    Token 2: # a#
    Token 3: # C#
    Token 4: # Lisp#
    
    Choose a token to use for the next iteration of text prediction: Lisp
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to write code with Emacs
    Route 1: Emacs is the most common way to write code with the
    Route 2: Emacs is the most common way to write code with a
    Route 3: Emacs is the most common way to write code with C
    Route 4: Emacs is the most common way to write code with Lisp
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to write code with Lisp#
    ###############################################################################
    
    

     90%|█████████ | 9/10 [01:19<00:08,  8.28s/it]

    
    Enter 'exit program' or 'Exit Program' to terminate the program.
    The top k=5 tokens are:
    Note: The '#' are there to delimit the start and end of the token since tokens can include '\n' and other invisible characters.
    Note: Type in the EXACT characters you see (or don't see), including whitespace, etc.
    
    Token 0: #.#
    Token 1: #,#
    Token 2: # and#
    Token 3: # in#
    Token 4: # (#
    
    Choose a token to use for the next iteration of text prediction:.
    
    ###############################################################################
    Original (excluding text prediction) raw text string: Emacs is
    
    All routes that user could have made in choosing a token from the current iteration:
    Route 0: Emacs is the most common way to write code with Lisp.
    Route 1: Emacs is the most common way to write code with Lisp,
    Route 2: Emacs is the most common way to write code with Lisp and
    Route 3: Emacs is the most common way to write code with Lisp in
    Route 4: Emacs is the most common way to write code with Lisp (
    Note: The '#' at the beginning and end delimit the start and end of the text.
    Predicted (excluding original raw input text) text string: # the most common way to write code with Lisp.#
    ###############################################################################
    
    

    100%|██████████| 10/10 [01:25<00:00,  8.53s/it]
    

    Iterations for current string has ended.  Will request user enter new string.
    
    Enter a string: exit
    Terminating program execution.
    
