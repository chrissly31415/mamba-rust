//! # Mamba-rs
//!
//! `Mamba-rs` is a tool for fast bond perception of molecules
//! using machine learned models.
//! Using ndarray (mathematical arrays) and polars (dataframe)
//!
//! Generation of training data:
//!
//! babel -m -h --gend3D -isdf ..\..\opera_data\OPERA_BP\TST_BP_1358.sdf -oxyz test.xyz

use std::{env, fs};
use std::error::Error;

use clap::{app_from_crate, arg};

mod mamba;
mod utils;
mod ml;

use mamba::{mol_from_xyz,  create_molblock};
use ml::{predict_mol, train_xgb, eval_xgb};


fn main() -> Result<(), Box<dyn Error>> {
    let snake = String::from_utf8(vec![0xF0, 0x9F, 0x90, 0x8D]).unwrap();
    println!("{} mamba-rs {}", snake, snake);
    let matches = app_from_crate!()
        .arg(arg!(-f --filename <NAME>).required(true))
        .override_help("xyz file")
        .arg(arg!(-v - -verbose))
        .get_matches();
    
    let filename = matches.value_of("filename").expect("required");

    if filename.len() > 1 {
        let mol = mol_from_xyz(filename).expect("Could not open file!");
        let df = predict_mol(&mol);
        println!("{}",df);
        let molblock = create_molblock(mol,df)?;
        let outfile = filename.replace(".xyz",".sdf");
        println!("Writing SD file:{}",outfile);
        fs::write(outfile, molblock).expect("Unable to write SD file");
    } else {
        train_xgb();
        eval_xgb("../mamba/libsvm_large.dat");
    }
    Ok(())
}
