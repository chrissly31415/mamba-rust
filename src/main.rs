//! # Mamba-rs
//!
//! `Mamba-rs` is a tool for fast bond perception of molecules
//! using machine learned models.
//! Using ndarray (mathematical arrays) and polars (dataframe)
//!
//! Generation of /testing data, e.g.
//!
//! babel -m -h --gend3D -isdf ..\..\opera_data\OPERA_BP\TST_BP_1358.sdf -oxyz test.xyz
//! Training data needs to be in libsvm format currently

use std::error::Error;
use std::{env, fs};

use clap::{Arg, ArgGroup,command};

use mambalib::ml::{eval_xgb, predict_mol};
use mambalib::{create_molblock, mol_from_xyz_file};

fn main() -> Result<(), Box<dyn Error>> {
    let snake = String::from_utf8(vec![0xF0, 0x9F, 0x90, 0x8D]).unwrap();
    println!("{} mamba-rs {}", snake, snake);

    let arguments = command!()
        .arg(
            Arg::new("filename")
                .short('f')
                .long("filename")
                .value_name("NAME")
                
        )
        .arg(
            Arg::new("train-dataset")
                .long("train")
                .value_name("TRAIN_DATASET")
                .requires("test-dataset"),
        )
        .arg(
            Arg::new("test-dataset")
                .long("test")
                .value_name("TEST_DATASET")
                .requires("train-dataset"),
        )
        .group(ArgGroup::new("datasets").args(&["train-dataset", "test-dataset"]))
        .arg(Arg::new("verbose").short('v').long("verbose"))
        .get_matches();


    if let Some(filename) = arguments.get_one::<String>("filename") { 
        let mol = mol_from_xyz_file(filename).expect("Could not open file!");
        let df = predict_mol(&mol);
        println!("{}", df);
        let molblock = create_molblock(mol, df)?;
        let outfile = filename.replace(".xyz", ".sdf");
        println!("Writing SD file:{}", outfile);
        fs::write(outfile, molblock).expect("Unable to write SD file");
    } else {
        // If you want to access the train and test datasets:
        if let Some(train_dataset) = arguments.get_one::<String>("train-dataset") {
            println!("Train dataset: {}", train_dataset);
            //train_xgb("../mamba/libsvm_large.dat","../mamba/3dqsar_test.dat");
            // Test dataset is optional, so use if let
            if let Some(test_dataset) = arguments.get_one::<String>("train-dataset") {
                println!("Test dataset: {}", test_dataset);
                eval_xgb(test_dataset);
                //eval_xgb("../mamba/libsvm_large.dat");
            }
        }    
    }
    Ok(())
}
