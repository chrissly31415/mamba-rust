use std::io;
use std::fs;
use std::num;
use std::mem;
use std::error::Error;

use ndarray::{Axis,Array,Array2,arr1,arr2,ArrayView1,indices_of};
use ndarray::{s,array};

use crate::utils;
use utils::{Float,argsort,distance_matrix};

const MAXAT: usize = 3;
const DIST_CUTOFF: Float = 3.0;
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
    let mol = parse_contents(&contents).expect(&format!("{} {}","Could parse file: ",filename));
    Ok(mol)
}

///Parsing the contents of an string derived from an xyz file
fn parse_contents(contents: &str) ->Result<Molecule,Box<Error>> {
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

/// Create polar dataframe with local bond information from distance matrix
/// 
///  q  ata  atb    distab  ata1    dista1   dista1b  ata2    dista2   dista2b  ata3    dista3   dista3b  atb1    distb1   distb1a  atb2    distb2   distb2a  atb3    distb3   distb3a
///  -1 7    6  1.452657     6  1.344503  2.465101     6  1.461155  2.429833     1  2.103436  1.101881     1  1.101881  2.103436     6  1.502192  2.478834     6  1.572784  2.352358
pub fn create_dataframe(mol: Molecule) ->Result<(),Box<Error>> {
    let dm  = distance_matrix(&mol.coords);
    println!("{}",dm);
    //iterate over rows of distance matrix
    println!("id1  id2  q  ata  atb    distab  ata1    dista1   dista1b  ata2    dista2   dista2b  ata3    dista3   dista3b  atb1    distb1   distb1a  atb2    distb2   distb2a  atb3    distb3   distb3a");
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
            //now go over neighbors of i an j
            for a in [i_tmp,j_tmp] {
                let b = if a==i_tmp {j_tmp} else {i_tmp};
                let row = dm.row(a).to_vec();
                let row_sorted = argsort(&row);
                let mut k = 0;
                for nextn in row_sorted.into_iter() {
                    if (nextn == j_tmp || nextn == i_tmp) {
                        continue
                    }                        
                    if k>=N_CUT {break};
                    let dist = dm[[a,nextn]];
                    let el_next = &mol.atoms[nextn];
                    let an_next = ELEMENTS.iter().position(|&s| s == el_next).unwrap_or_default();
                    let distb = dm[[b,nextn]];
                    print!(" {:3} {:.4} {:.4}",an_next,dist, distb);
                    k=k+1;
                }


            }
            println!();
        }
   
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parse_xyz() {
        let mol = mol_from_xyz("data\\test1.xyz").expect("Could not open file!");
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
        let mol = mol_from_xyz("data\\test1.xyz").expect("Could not open file!");
        create_dataframe(mol);
    }
}