use std::io;
use std::fs;
use std::num;

use ndarray::{Axis,Array,Array2,arr1,arr2,ArrayView1,indices_of};
use ndarray::{s,array};

use crate::utils;
use utils::{Float,argsort,distance_matrix};

const MAXAT: usize = 3;

/// Simple Molecule structure
pub struct Molecule {
    pub natoms: usize,
    pub atoms: Vec<String>,
    pub coords: Array2<Float>,
    pub info: String,
    
}

/// Implementation of Molecule structure
impl Molecule {
    /// creating a molecule from its core features 
    pub fn create_molecule(natoms: usize, atoms: Vec<String>, coords: Array2<Float>, info: String) -> Molecule {
        Molecule {    
            natoms,       
            atoms,
            coords,
            info
        }
    }
}


/// Read the contents of a file to a string
pub fn mol_from_xyz(filename: &str) -> Result<Molecule, io::Error> {
    let contents = fs::read_to_string(filename)?;
    let mol = parse_contents(&contents).expect(&format!("{} {}","Could parse file: ",filename));
    Ok(mol)
}

///Parsing the contents of an string derived from an xyz file
fn parse_contents(contents: &str) ->Result<Molecule,num::ParseFloatError> {
    let lines = contents.split("\n");
    let mut atoms: Vec<String> = Vec::new();
    let mut coords: Vec<Float> = Vec::new();
    let mut nrows = 0;
    let mut natoms : Float = 0.0;
    let mut info: &str = "";
    for (i,line) in lines.enumerate() {
        
        if i==0 {
             natoms = line.trim().parse()?;
        }
        if i==1 {
            info  = line;
        }
        if i>1 && line.trim().len()>0 {
            nrows +=1;
            let mut iter = line.split_whitespace();
            let atom: &str = iter.next().unwrap_or_default();
            atoms.push(atom.to_owned());
            let x: Float = iter.next().unwrap_or_default().parse()?;
            let y: Float = iter.next().unwrap_or_default().parse()?;
            let z: Float = iter.next().unwrap_or_default().parse()?;
            let mut coord = vec![x, y, z];
            coords.append(&mut coord);
        }
    }
    assert_eq!(natoms as usize,nrows);
    let coords = Array2::from_shape_vec((nrows, 3), coords).unwrap();
    let molecule = Molecule::create_molecule(natoms as usize, atoms, coords, info.to_owned());
    Ok(molecule)
}

/// Create polar dataframe with local bond information from distance matrix
pub fn create_dataframe(mol: Molecule) {
    let dm  = distance_matrix(&mol.coords);
    println!("{}",dm);
    //iterate over rows of distance matrix
    for (i,r) in dm.axis_iter(Axis(0)).enumerate() {
        let row = r.to_vec();
        let row_sorted = argsort(&row);
        let idx_center = row_sorted[0];
        let atom_center = &mol.atoms[idx_center];
        let slice = &row_sorted[1..];
        print!("{:4} {}",i,atom_center);
        for (j,idx_next) in slice.iter().cloned().enumerate() {
            let atom_next = &mol.atoms[idx_next];
            let dist_next = dm[[idx_center,idx_next]];
            print!(" {} d: {:.2}",atom_next,dist_next);
            if j>MAXAT {
                break;
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_xyz() {
        let mol = mol_from_xyz("data\\compound1.xyz").expect("Could not open file!");
        println!("coords: {:?}", mol.coords);
        println!("mol: {:?}", mol.atoms);
        //distance_matrix(&mol.coords);
    }
    #[test]
    fn parse_all() {
        let entries = fs::read_dir("./data").unwrap();
        for (i,entry) in entries.enumerate() {
            let path = entry.unwrap().path();
            let ext = path.extension().unwrap().to_str().unwrap();
            if path.is_file() && ext == "xyz" {
                let fname = path.to_str().unwrap();
                let mol = mol_from_xyz(fname).expect("Could not read file!");
                distance_matrix(&mol.coords);
                //println!("i: {} natoms: {} Name: {} ",i,mol.natoms,path.display());
            }
        }
    }
    #[test]
    fn test_df() {
        let mol = mol_from_xyz("data\\test37.xyz").expect("Could not open file!");
        create_dataframe(mol);
    }
}