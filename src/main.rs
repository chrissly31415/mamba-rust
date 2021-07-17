//! # Mamba-rs
//! 
//! `Mamba-rs` is a tool for fast bond perception of molecules
//! using machine learned models.
//! Using ndarray (mathematical arrays) and polars (dataframe)
//! 
//! Generation of training data:
//! 
//! babel -m -h --gend3D -isdf ..\..\opera_data\OPERA_BP\TST_BP_1358.sdf -oxyz test.xyz

mod mamba;
mod utils;

use std::env;

use mamba::{create_dataframe,mol_from_xyz};

fn main() {
    println!("mamba-rs version 0.1");
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
    if args.len()>1 {
        let mol = mol_from_xyz(&args[1]).expect("Could not open file!");
        create_dataframe(mol);
    }
}
