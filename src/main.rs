use std::collections::HashMap;
use std::env;
use deltaml::{classical_ml::{Algorithm, algorithms::RandomForest, losses::CrossEntropy}, ndarray::{Array1, Array2}};
use polars::prelude::*;

/* Encoding Map
* Iris-setosa       0
* Iris-versicolor   1
* Iris-virginica    2
*/

fn get_dataset_path()-> String{
    // Get the current directory (relative to the Cargo.toml)
    #[cfg(windows)]
    {
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let dataset_path = format!("{}\\..\\dataset\\iris-rs.csv", manifest_dir);
        dataset_path
    }
    #[cfg(not(windows))]
    {
        let dataset_path = "iris-rs.csv".to_string();
        dataset_path
    }
}

fn main() -> Result<(), PolarsError>{

    let map_to_name: HashMap<u8, &str> = HashMap::from([
        (0, "Iris-setosa"),
        (1, "Iris-versicolor"),
        (2, "Iris-virginica"),
    ]);

    let raw_dataset_df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(get_dataset_path().into()))?
        .finish()?;

    let sample_size = (raw_dataset_df.shape().0 as f64 * 0.8).trunc() as usize;
    let sample_dataset_df = raw_dataset_df.sample_n_literal(sample_size, false, true, Some(1337u64))?;

    let y_data = Array1::from_iter(sample_dataset_df
                                       .column("Species")?
                                       .to_owned()
                                       .into_frame()
                                       .to_ndarray::<Float64Type>(Default::default())?);

    // Drop "species" column to get features dataframe and convert to ndarray
    let x_data = sample_dataset_df
        .drop("Species")?
        .to_ndarray::<Float64Type>(Default::default())?;

    // Init Model
    let mut model = RandomForest::new(CrossEntropy);

    // Train the model
    let learning_rate = 0.01;
    let epochs = 1000;
    model.fit(&x_data, &y_data, learning_rate, epochs);

    // Test point to new data array
    let new_x_data = Array2::from_shape_vec((1, 4), vec![5.1,3.4,1.4,0.2]).unwrap();

    // Make a prediction on the new data array
    let prediction = model.predict(&new_x_data);

    // Output array to vec
    let prediction_vec = prediction.to_vec();
    let prediction_value = prediction_vec[0]; // Our model only outputs 1 prediction

    // Get the name using .get()
    if let Some(species_name) = map_to_name.get(&((&prediction_value).trunc() as u8)) {
        println!("Predicted Species: {}", species_name);
    } else {
        println!("No matching species found for prediction: {}", prediction_value);
    }

    Ok(())
}
