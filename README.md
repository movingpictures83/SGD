# SGD
# Language: Python
# Input: TXT
# Output: TXT
# Tested with: PluMA 1.1, Python 3.6

PluMA plugin that runs Adam optimization (Kingma and Ba, 2014)

The plugin expectes as input a tab-delimited file of keyword-value pairs:
inputfile: Dataset
lr: Learning Rate
epochs: Number of epochs to run
start: Startin column
stop: Ending column

Output is sent to a TXT file
