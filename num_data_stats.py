# @title Setup - Import relevant modules

# The following code imports relevant modules that
# allow you to run the colab.
# If you encounter technical issues running some of the code sections
# that follow, try running this section again.

import pandas as pd

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

#@title Import the dataset

# The following code imports the dataset that is used in the colab.

training_df = pd.read_csv(filepath_or_buffer="https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv")

# Get statistics on the dataset.

# The following code returns basic statistics about the data in the dataframe.

training_df.describe()
