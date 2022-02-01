# Neural_Network_Charity_Analysis

## Overview

We are working on a model to predict where to make investments. We are using machine learning and neural networks to classify the foundation.

## Results:

### Data Preprocessing

First we import the data from a dataset to create a dataframe to manage the info.

    # Import our dependencies
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler,OneHotEncoder
    import pandas as pd
    import tensorflow as tf

    #  Import and read the charity_data.csv.
    import pandas as pd 
    application_df = pd.read_csv("Resources/charity_data.csv")
    application_df.head()

![image](https://user-images.githubusercontent.com/88845919/151902118-8f44fc54-dbcf-4da2-ac7a-a67ede7d3d0e.png)

For all this variables we needed to clean the information. We start eliminating some columns, 'EIN' and 'NAME'.

    # Drop the non-beneficial ID columns, 'EIN' and 'NAME'.
    application_df.drop(['EIN','NAME'],1, inplace =True)
    application_df.head()

![image](https://user-images.githubusercontent.com/88845919/151902250-193d8b1f-dd71-4874-9f57-d95cf3db05bd.png)

We made some research of the features that we have. Specially to 'APPLICATION_TYPE' and 'CLASSIFICATION'. This help us to create and 'other' category to manage easier the info.

This let us see in a graph the date with the density and determine the limite for the 'other' category.

![image](https://user-images.githubusercontent.com/88845919/151902579-6ec38ef6-7e95-4b7b-80a4-690d19fdbfae.png)

Then we replace the data.

![image](https://user-images.githubusercontent.com/88845919/151902626-56a2354b-68a6-4e79-b735-e0bfb43465f7.png)

Same for 'CLASSIFICATION'.

![image](https://user-images.githubusercontent.com/88845919/151902675-df00753b-3d59-4408-9bea-9dc6d601ed51.png)

![image](https://user-images.githubusercontent.com/88845919/151902706-15279c66-5182-4626-be50-d23aa50a3f99.png)

- What variable(s) are considered the target(s) for your model?

As target I consider the variable "IS_SUCCESSFUL", to determine if the program work or not and check if it's a good option to invest.

    # Split our preprocessed data into our features and target arrays
    y = application_df["IS_SUCCESSFUL"].values
    X = application_df.drop(["IS_SUCCESSFUL"],1).values

- What variable(s) are considered to be the features for your model?

Fort the features at first I select all the remaining features, but I needed to change the categorical variables to numerical data. So I use OneHotEncoder.

![image](https://user-images.githubusercontent.com/88845919/151902982-0e80b4f3-743e-47ee-802c-cd172f52ef57.png)


- What variable(s) are neither targets nor features, and should be removed from the input data?

For optimization, I elimite the following features.

    application_df.drop(['SPECIAL_CONSIDERATIONS', 'STATUS', 'ASK_AMT', 'USE_CASE'], 1, inplace=True)

### Compiling, Training, and Evaluating the Model

How many neurons, layers, and activation functions did you select for your neural network model, and why?

For the model I used 3 neurons with 90 and 80 layers. I made several attempts and follow the changes to check the influence of the changes and this looked as the better result I can get. The same process was to determine the usage of relu activation function. I also change the epochs.

    # Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
    number_input_features = len(X_train[0])
    hidden_nodes_layer1 = 90
    hidden_nodes_layer2 = 80
    hidden_nodes_layer3 = 80
    #hidden_nodes_layer4 = 80

    nn = tf.keras.models.Sequential()

    # First hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))


    # Second hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="relu"))

    # Third hidden layer
    nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="relu"))

    # Third hidden layer
    #nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer4, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="relu"))

    # Check the structure of the model
    nn.summary()

Were you able to achieve the target model performance?

No, the higher performance I could get was 0.72688.

![image](https://user-images.githubusercontent.com/88845919/151903418-7a31c873-a55a-4fd3-ae36-27b7d77cb587.png)

What steps did you take to try and increase model performance?

I tried eliminating different features and checking the influence in the model. Adjust and accumulate better results. Then I add and remove neurons and layers, and try to find some balance with different activations functions. 

Summary:

In general the higher performance i could get was 0.72688. This using deep learning model. I made several changes on the input information, in the model, like neurons, layers and activation functions. Also in epochs. It seems that compering to other models we can use the random forest to make the analysis faster. They both have similar performance, but we can adjust easier using this new model.
