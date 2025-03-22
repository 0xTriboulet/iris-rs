# Iris Classification with Random Forest in Rust
This project demonstrates how to build a **Random Forest classifier** using the [Delta](https://github.com/blackportal-ai/delta) library to classify the Iris dataset. The project leverages core Rust features and external libraries such as `Polars` for data manipulation and `Delta` for machine learning.
## Overview
The project uses the [Iris dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) and trains a **Random Forest classifier** to predict the species of Iris flowers based on several features. The trained model predicts the species of a new sample and outputs the result in a human-readable format.
Key components of the project:
- **Data preprocessing**: Reading the CSV dataset and converting it into an ndarray format using `Polars`.
- **Model training**: Training a Random Forest model using Delta's `classical_ml` module.
- **Prediction**: Making predictions on new input data and mapping numeric labels to species names.

## Requirements
- **Rust installation**: Ensure you have Rust installed. You can get it from [rustup.rs](https://rustup.rs/).
- **Dependencies**:
    - [Delta](https://github.com/blackportal-ai/delta) (ML library for Rust)
    - [Polars](https://github.com/pola-rs/polars) (DataFrame library for Rust)
    - [ndarray](https://docs.rs/ndarray) (Multidimensional array support)

To configure the project's toolchain, the file `rust-toolchain.toml` specifies usage of `stable-x86_64-pc-windows-gnu`.
## Installation & Setup
1. **Clone the repository**:
``` bash
   git clone https://github.com/<your-repo>/iris-rs.git
   cd iris-rs
```
1. **Install Rust and the specified toolchain**: Ensure you have the `stable-x86_64-pc-windows-gnu` toolchain installed. If not, install it via:
``` bash
   rustup install stable-x86_64-pc-windows-gnu
   rustup override set stable-x86_64-pc-windows-gnu
```
1. **Install dependencies**: Run the following command to fetch and install the project's dependencies:
``` bash
   cargo build
```
1. **Dataset**:
    - Place the Iris dataset CSV (`iris.csv`) in your desired location.
    - Update the `DATASET_PATH` constant in `main.rs` with the path to the CSV file.

Example:
``` rust
   static DATASET_PATH: &str = r"C:\path\to\iris.csv";
```
1. **Run the project**: To build and execute the project, use:
``` bash
   cargo run --release
```
## Project Structure
- `main.rs` : The main entry point of the project. It includes:
    - Data loading and preprocessing using Polars.
    - A Random Forest model implementation with the `Delta` library.
    - Mapping of predictions to Iris species names using `HashMap`.

- `rust-toolchain.toml` : Specifies the Rust toolchain to use (e.g., `stable-x86_64-pc-windows-gnu`).
- `Cargo.toml` : Lists the dependencies and configurations for the project.

## Features
1. **Data Handling**:
    - Use the `Polars` library to load and manipulate the Iris dataset.
    - Split the dataset into an 80% training and 20% testing subset.

2. **RandomForest Classifier**:
    - Train a Random Forest model on the Iris dataset using `DeltaML`.
    - Use `CrossEntropy` loss for multi-class classification.

3. **Prediction**:
    - Predict the species of a flower using a new input array (e.g., `[5.1, 3.5, 1.4, 0.2]`).
    - Output the predicted species in a user-friendly format:
``` bash
     Predicted Species: Iris-setosa
```
## Dependencies
The project relies on the following crates:
- `deltaml`: Machine learning library (`classical_ml` module for Random Forest support). [GitHub link](https://github.com/blackportal-ai/delta).
- `polars` & `polars-core`: High-performance DataFrame library for Rust.
- `ndarray`: Library for working with n-dimensional arrays.

## Notes

- This example leaves a lot of room for optimizations. The categorical labels in the Iris dataset were pre-processed to minimize the data manipulation steps performed in Rust. Particularly, easily encoding a column did not appear to be a capability in `Polars`.
- Using the `Delta` library provides  the ability to create stand-alone executables with ML primitives. Building with the `*-pc-windows-gnu` toolchains maximizes compatability with reflective loading.