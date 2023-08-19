extern crate xgboost;
use std::fs;

use xgboost::{parameters, Booster, DMatrix};

use polars::prelude::*;

use crate::{
    create_dataframe,
    utils::{accuracy, df2vec},
    XYZMolecule,
};

pub fn train_xgb(trainpath: &str, testpath: &str) {
    //this function needs libsvm data sets
    //let dtrain = DMatrix::load("../mamba/3dqsar_train.dat").unwrap();
    let dtrain = DMatrix::load(trainpath).unwrap();
    println!("Train matrix: {}x{}", dtrain.num_rows(), dtrain.num_cols());
    let dtest = DMatrix::load(testpath).unwrap();
    println!("Test matrix: {}x{}", dtest.num_rows(), dtest.num_cols());

    // configure objectives, metrics, etc.
    let learning_params = parameters::learning::LearningTaskParametersBuilder::default()
        .objective(parameters::learning::Objective::MultiSoftmax(5))
        .build()
        .unwrap();

    // configure the tree-based learning model's parameters
    let tree_params = parameters::tree::TreeBoosterParametersBuilder::default()
        .max_depth(6)
        .eta(0.1)
        .build()
        .unwrap();

    // overall configuration for Booster
    let booster_params = parameters::BoosterParametersBuilder::default()
        .booster_type(parameters::BoosterType::Tree(tree_params))
        .learning_params(learning_params)
        .verbose(false)
        .build()
        .unwrap();
    // specify datasets to evaluate against during training
    let evaluation_sets = [(&dtest, "test"), (&dtrain, "train")];
    // overall configuration for training/evaluation
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain) // dataset to train with
        .boost_rounds(200) // number of training iterations
        .booster_params(booster_params) // model parameters
        .evaluation_sets(Some(&evaluation_sets)) // optional datasets to evaluate against in each iteration
        .build()
        .unwrap();

    // train booster model, and print evaluation metrics
    println!("\nTraining tree booster...");
    let booster = Booster::train(&training_params).unwrap();

    // save and load model file
    println!("\nSaving and loading Booster model...");
    booster.save("xgb.model").unwrap();

    // get predictions probabilities for given matrix
    let preds = booster.predict(&dtrain).unwrap();

    // get predicted labels for each test example (i.e. 0 or 1)
    println!("\nChecking predictions...");
    let labels = dtrain.get_labels().unwrap();
    for (t, p) in labels.iter().zip(preds.iter()) {
        println!("t:{} p:{}", t, p);
    }

    let acc = accuracy(&preds, labels);
    println!(
        "accuracy={}% ({}/{} correct)",
        acc,
        acc * preds.len() as f32,
        preds.len()
    );
}

pub fn eval_xgb(evaldata: &str) {
    println!("\nLoading eval data set...");
    let dtest = DMatrix::load(evaldata).unwrap();
    println!("\nLoading xgb model...");
    let booster = Booster::load("xgb.model").unwrap();
    // get predictions probabilities for given matrix
    let preds = booster.predict(&dtest).unwrap();

    // get predicted labels for each test example (i.e. 0 or 1)
    let labels = dtest.get_labels().unwrap();

    let acc = accuracy(&preds, labels);
    println!(
        "accuracy={}% ({}/{} correct)",
        acc,
        acc * preds.len() as f32,
        preds.len()
    );
}

pub fn predict_mol(mol: &XYZMolecule) -> DataFrame {
    let df = create_dataframe(mol).unwrap();

    let model = "xgb.model";
    println!("Loading xgb-model:{}", model);
    let booster = Booster::load(model).unwrap();

    let flat_vec = df2vec(&df);

    let (n, _) = df.shape();
    let dtest = DMatrix::from_dense(&flat_vec, n).unwrap();
    let preds = Series::new("preds", booster.predict(&dtest).unwrap());

    let df = df.hstack(&[preds]).unwrap();

    let file = fs::File::create("df.csv").expect("could not create file");
    CsvWriter::new(&file)
        .has_header(true)
        .with_delimiter(b',')
        .finish(&df)
        .unwrap();
    df
}
