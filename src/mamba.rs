#![feature(str_split_whitespace_as_str)]

use std::fs;
use std::mem;
use std::error::Error;
use std::result::Result;

use ndarray::{Axis,Array,Array2,arr1,arr2,ArrayView1,indices_of};

use polars::prelude::*;

use crate::utils;
use utils::{argsort,distance_matrix, transpose};

/// float type can be change
pub type Float = f32;
/// distance cut off
const DIST_CUTOFF: Float = 3.0;
/// number of neighbors
const N_CUT: usize = 3;

static ELEMENTS: &[&str] = &["X","H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I"];

/// Simple Molecule structure
#[derive(Default)]
pub struct Molecule {
    pub natoms: usize,
    pub atoms: Vec<String>,
    pub coords: Array2<Float>,
    pub q: i32,
    pub info: String, 
}

/// Implementation of Molecule structure
impl Molecule {
    /// creating a molecule from its core features 
    pub fn new(natoms: usize, atoms: Vec<String>, coords: Array2<Float>, q: i32,  info: String ) -> Self {
        Molecule {    
            natoms,       
            atoms,
            coords,
            q,
            info,
        }
    }
}

/// Read the contents of a file to a string
pub fn mol_from_xyz(filename: &str) -> Result<Molecule, Box<Error>> {
    let contents = fs::read_to_string(filename)?;
    let mol = parse_contents(&contents)?;
    Ok(mol)
}

///Parsing the contents of an string derived from an xyz file
fn parse_contents(contents: &str) ->Result<Molecule,Box<dyn Error>> {
    let lines = contents.split("\n");
    let mut atoms: Vec<String> = Vec::new();
    let mut coords: Vec<Float> = Vec::new();
    let mut nrows = 0;
    let mut natoms : usize =0;
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
    let molecule = Molecule::new(natoms as usize, atoms, coords, 0,info.to_owned());
    Ok(molecule)
}


/// Create a 2D ndarray with local bond information from distance matrix
/// https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html#similarities
pub fn create_dataframe(mol: Molecule) ->Result<(),Box<dyn Error>> {
    let dm  = distance_matrix(&mol.coords);
    assert_eq!(mol.natoms,dm.ncols());
    //let mut features = Array2::from_elem((mol.natoms, 22),0.0);
    //let header = "id1  id2  q  ata  atb    distab  ata1    dista1   dista1b  ata2    dista2   dista2b  ata3    dista3   dista3b  atb1    distb1   distb1a  atb2    distb2   distb2a  atb3    distb3   distb3a";
    //let col:  Vec<&str> = header.split_whitespace().collect();
    let mut header = Vec::<&str>::new();
    //let ncol = col.len();
    let mut features = Vec::<Vec<Float>>::new();
    println!("first:{:?}",features);
    //iterate over rows of distance matrix
    println!();
    for i in 0..dm.ncols() {
        for j in 0..dm.ncols() {
            if i>=j {
                continue;
            }
            let dist = dm[[i,j]];
            if dist>DIST_CUTOFF {
                continue;
            }
            
            let mut i_tmp = i;
            let mut j_tmp = j;
            let e1 = &mol.atoms[i_tmp];
            let e2 = &mol.atoms[j_tmp];
            let mut an1 = ELEMENTS.iter().position(|&s| s == e1).unwrap_or_default();
            let mut an2 = ELEMENTS.iter().position(|&s| s == e2).unwrap_or_default();
            if an1<an2 {
                mem::swap(&mut i_tmp,&mut j_tmp); 
                mem::swap(&mut an1, &mut an2);
            } 
            print!(" {:2} {:2} {:2} {:2} {:2} {:.4}",i_tmp+1,j_tmp+1,mol.q,an1,an2,dist);
            
            //assign to 2d array
            let mut data_row = vec![];
            data_row.push(i_tmp as Float + 1.0);
            data_row.push(j_tmp as Float + 1.0);
            data_row.push(mol.q as Float);
            data_row.push(an1 as Float);
            data_row.push(an2 as Float);
            data_row.push(dist);
            if features.len()==0 {
                header.append(&mut vec!["id1","id2","q","ata","atb","distab"]);
            }

            //now go over neighbors of i an j
            for a in [i_tmp,j_tmp] {
                let b = if a==i_tmp {j_tmp} else {i_tmp};
                let row = dm.row(a).to_vec();
                let row_sorted = argsort(&row);
                let mut k = 0;
                for nextn in row_sorted.into_iter() {
                    if nextn == j_tmp || nextn == i_tmp {
                        continue
                    }                        
                    if k>=N_CUT {break};
                    let dist = dm[[a,nextn]];
                    let el_next = &mol.atoms[nextn];
                    let an_next = ELEMENTS.iter().position(|&s| s == el_next).unwrap_or_default();
                    let distb = dm[[b,nextn]];
                    print!(" {:3} {:.4} {:.4}",an_next,dist, distb);
                    data_row.push(an_next as Float);
                    data_row.push(dist);
                    data_row.push(distb);
                    if features.len()==0 { 
                        let a: String = format!("at{:}",k);
                        let b = a.to_owned().as_str();
                        header.append(&mut vec![b,"dista","distb"]);
                    }
                    k +=1;
                }
            }
            features.push(data_row);
            println!();
        }
    }
    //transpose data first...
    let features_col = transpose(features);

    println!("{:?}",features_col);
    //println!("{:?}",&header);
    let mut series = Vec::<Series>::new();
    assert_eq!(features_col.len(),header.len());
    let zip_iter = features_col.iter().zip(header.iter());
    for (col,name) in zip_iter {
        let  s = Series::new(name, col.to_owned()); 
        series.push(s);
    }
    let df = DataFrame::new(series).unwrap();

    println!("{:?}",df);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_xyz() {
        let mol = mol_from_xyz("data\\test1.xyz").expect("Could not open file!");
        println!("coords: {:?}", mol.coords);
        assert_eq!(mol.coords.len(),69);
        println!("mol: {:?}", mol.atoms);
        assert_eq!(mol.atoms.len(),23);
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
        let mol = mol_from_xyz("data\\test1.xyz").expect("Could not open file!");
        create_dataframe(mol);
    }
}