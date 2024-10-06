import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from scipy.special import expit  # sigmoid function
import random
import pickle

# Function to initialize ELM weights and biases
def initialize_elm(input_size, hidden_size):
    weights = np.random.uniform(-1, 1, (input_size, hidden_size))  # Input to hidden layer weights
    biases = np.random.uniform(-1, 1, (hidden_size,))  # Biases for hidden layer neurons
    return weights, biases

# ELM forward propagation
def elm_forward(X, weights, biases):
    H = np.dot(X, weights) + biases  # Linear combination
    H = expit(H)  # Activation using sigmoid
    return H

# Function to calculate output weights (beta)
def calculate_output_weights(H, Y):
    beta = np.dot(np.linalg.pinv(H), Y)  # Moore-Penrose pseudo-inverse
    return beta

# ELM model prediction
def elm_predict(X, input_weights, biases, output_weights):
    H = elm_forward(X, input_weights, biases)
    Y_pred = np.dot(H, output_weights)
    return Y_pred

# Genetic Algorithm fitness function: MAPE
def fitness_function(weights, biases, X_train, Y_train, hidden_size):
    H = elm_forward(X_train, weights.reshape(-1, hidden_size), biases)
    output_weights = calculate_output_weights(H, Y_train)
    Y_pred = elm_predict(X_train, weights, biases, output_weights)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = mean_absolute_percentage_error(Y_train, Y_pred)
    return mape

# Genetic Algorithm to optimize ELM weights
def genetic_algorithm(X_train, Y_train, input_size, hidden_size, pop_size, generations):
    # Initialize population
    population = [initialize_elm(input_size, hidden_size) for _ in range(pop_size)]
    
    # Lists to store the best weights and MAPE for each generation
    all_best_weights = []
    all_best_mapes = []

    for generation in range(generations):
        # Evaluate fitness of population
        fitness_scores = []
        for weights, biases in population:
            fitness = fitness_function(weights, biases, X_train, Y_train, hidden_size)
            fitness_scores.append(fitness)
        
        # Sort population based on fitness (lower MAPE is better)
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0])]
        
        # Select parents (elitism)
        parents = sorted_population[:pop_size//2]
        
        # Crossover and mutation to generate new population
        new_population = parents[:]
        for i in range(len(parents) // 2):
            parent1_weights, parent1_biases = parents[i]
            parent2_weights, parent2_biases = parents[-(i+1)]
            
            # Crossover
            child1_weights = (parent1_weights + parent2_weights) / 2
            child1_biases = (parent1_biases + parent2_biases) / 2
            child2_weights = (parent1_weights + parent2_weights) / 2
            child2_biases = (parent1_biases + parent2_biases) / 2
            
            # Mutation
            mutation_rate = 0.1
            if random.random() < mutation_rate:
                child1_weights += np.random.uniform(-0.5, 0.5, child1_weights.shape)
                child1_biases += np.random.uniform(-0.5, 0.5, child1_biases.shape)
            if random.random() < mutation_rate:
                child2_weights += np.random.uniform(-0.5, 0.5, child2_weights.shape)
                child2_biases += np.random.uniform(-0.5, 0.5, child2_biases.shape)
            
            new_population.append((child1_weights, child1_biases))
            new_population.append((child2_weights, child2_biases))
        
        population = new_population[:pop_size]

        # Save the best weights and MAPE of this generation
        best_weights, best_biases = sorted_population[0]
        all_best_weights.append(best_weights)
        all_best_mapes.append(fitness_scores[0])

    # Return best weights and biases from the final population, and the lists of best weights and MAPE
    return best_weights, best_biases, all_best_weights, all_best_mapes

# Fungsi untuk menghitung hidden layer ELM
def hidden_layer(X, W, b):
    return np.tanh(np.dot(X, W) + b)

# Fungsi untuk melatih ELM
def fit_elm(X, y, n_hidden_units):
    n_samples, n_features = X.shape
    W = np.random.rand(n_features, n_hidden_units)
    b = np.random.rand(1, n_hidden_units)

    H = hidden_layer(X, W, b)
    alpha = np.linalg.pinv(H).dot(y)

    return W, b, alpha

# Fungsi untuk memprediksi ELM
def predict_elm(X, W, b, alpha):
    H = hidden_layer(X, W, b)
    return H.dot(alpha)

st.set_page_config(layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>Prediksi Produksi Padi Berdasarkan Curah Hujan</h1>", unsafe_allow_html=True
)

with st.sidebar:
    selected = option_menu("Main Menu", ["Dataset", 'Modelling', 'Prediction'], default_index=2)

# load dataset 
data = pd.read_excel('dataset.xlsx')

# split dataset 
fitur = data.drop(['Kecamatan','Tahun', 'Produksi (Kw)'], axis=1)
target = data['Produksi (Kw)']
x_train, x_test, y_train, y_test = train_test_split(fitur, target, test_size=0.2, random_state=0)

# normalisasi
minmax = MinMaxScaler()
x_train_norm = minmax.fit_transform(x_train)
x_test_norm = minmax.transform(x_test)

if (selected == 'Dataset'):
    st.info("""
    Adapun tahapan - tahapan yang akan dilakukan pada persiapan data ini adalah :
    1. Tahap pengumpulan data
    4. Tahap split data
    5. Tahap normalisasi data
    """)
    
    # Tahap pengumpulan data
    st.subheader('Data Asli')
    st.write(data)

    # Split data
    st.title('Split Data --------------')
    st.write('Total data : ', fitur.shape[0])
    st.write('Total data latih   : ', x_train.shape[0])
    st.write('Total data testing : ', x_test.shape[0])

    # Normalisasi data
    st.title('Normalisasi Data ----------')
    # Convert the normalized arrays back to DataFrames for display
    x_train_norm_df = pd.DataFrame(x_train_norm, columns=[f'{col} (normalized)' for col in x_train.columns])
    x_test_norm_df = pd.DataFrame(x_test_norm, columns=[f'{col} (normalized)' for col in x_test.columns])

    # Display normalized data
    combined_train = pd.concat([x_train, x_train_norm_df], axis=1)
    st.subheader('Data Latih Sebelum dan Setelah Normalisasi')
    st.write(combined_train)

    combined_test = pd.concat([x_test, x_test_norm_df], axis=1)
    st.subheader('Data Testing Sebelum dan Setelah Normalisasi')
    st.write(combined_test)

if (selected == 'Modelling'):
    
    pop_size = 50
    crossover_rate = 0.5
    mutation_rate = 0.01
    hidden_units = 10 
    generation = 100

    st.info('Pencarian bobot terbaik menggunakan Algoritma Genetika :')
    st.write('Ukuran populasi = ', pop_size)
    st.write('Crossover = ', crossover_rate)
    st.write('Mutasi = ', mutation_rate)
    st.write('Generasi = ', generation)

    best_weights, best_biases, all_best_weights, all_best_mapes = genetic_algorithm(x_train_norm, y_train.values, x_train_norm.shape[1], hidden_units, pop_size, generations=generation)

    st.write("Bobot terbaik yang diperoleh:", best_weights)
    st.write("MAPE terbaik yang diperoleh:", all_best_mapes[0])

    st.title("Training Dataset ----------")
    for i in range(len(all_best_weights)):
        st.write(f"Generasi {i + 1}: Bobot = {all_best_weights[i]}, MAPE = {all_best_mapes[i]}")


    st.title("Testing Dataset ----------")
    # Implementasi bobot pada metode ELM
    W, b, alpha = fit_elm(x_train_norm, y_train.values, hidden_units)
    y_pred = predict_elm(x_test_norm, W, b, alpha)

    test_mape = mean_absolute_percentage_error(y_test, y_pred)
    st.write("MAPE yang diperoleh:", test_mape)

    results_df = pd.DataFrame({
    'Produksi Asli (Kw)': y_test,
    'Produksi Prediksi (Kw)': y_pred
    })

    st.subheader("Data Testing Asli vs Prediksi")
    st.write(results_df)

    # Save model parameters to a pickle file
    model_params = {
        'W': W,
        'b': b,
        'alpha': alpha,
        'mape': test_mape
    }
    with open('elm_model_params.pkl', 'wb') as f:
        pickle.dump(model_params, f)

# Add a new section to load the model from pickle if needed
if selected == 'Prediction':
    with open('elm_model_params.pkl', 'rb') as f:
        model_params = pickle.load(f)
        W_loaded = model_params['W']
        b_loaded = model_params['b']
        alpha_loaded = model_params['alpha']
        
    # Prediction section
    st.subheader("Make a Prediction")
    
    # Input field for "Curah Hujan (mm)"
    curah_hujan = st.number_input("Input Curah Hujan (mm)", value=0.0)

    if st.button('Predict'):
        # Prepare input data for prediction
        input_data = np.array([[curah_hujan]])  # Using only "Curah Hujan (mm)"

        # Normalize input data
        input_data_norm = minmax.transform(input_data)

        # Make prediction using the loaded model
        prediction = predict_elm(input_data_norm, W_loaded, b_loaded, alpha_loaded)
        
        # Display the prediction result
        st.write("Prediksi produksi padi (kw) : ")
        if prediction[0] < 0:
            st.danger(prediction[0])
        else:
            st.success(prediction[0])
