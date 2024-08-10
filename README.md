Some initial LSTMs to work on surrogate models for ExaEpi.

main.py has the main logic to build and train a model, with various options available by commenting/uncommenting lines.

data.py process the data

models.py creates the models

train.py trains the models

analysis.py can be used to see the results of the model (currently by default plots and saves all of the test examples).

All output is saved with the timestamp at the beginning of main.py as a suffix, including:

- model checkpoint with all of its learned weights
- analysis plots

Highest priority of work to be done is likely:
1. Training a one-to-fifty or one-to-one model with longer epochs
2. Figuring out a better normalization scheme than dividing by 1000

Other areas of interest are written at the top of files.

Feel free to email at arnav_singhal at brown dot edu or open an issue with questions or bugs.
