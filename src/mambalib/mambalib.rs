//#![feature(str_split_whitespace_as_str)]

use std::error::Error;
use std::fs;
use std::mem;
use std::path::PathBuf;
use std::result::Result;

use ml::predict_mol;
use ndarray::{ arr2, indices_of, Array, Array2};

use polars::prelude::*;

pub mod ml;
mod utils;

use utils::{argsort, distance_matrix, transpose};

/// float type can be change
pub type Float = f32;
/// distance cut off
const DIST_CUTOFF: Float = 3.0;
/// number of neighbors
const N_CUT: usize = 3;

static ELEMENTS: &[&str] = &[
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As",
    "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
    "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

/// Simple Molecule structure
#[derive(Default)]
pub struct XYZMolecule {
    pub natoms: usize,
    pub atoms: Vec<String>,
    pub coords: Array2<Float>,
    pub q: i32,
    pub info: String,
    pub name: String,
}

/// Implementation of Molecule structure
impl XYZMolecule {
    /// creating a molecule from its core features
    pub fn new(atoms: Vec<String>, coords: Array2<Float>, q: i32) -> Self {
        let natoms = atoms.len();
        assert_eq!(atoms.len(), coords.len() / 3);
        XYZMolecule {
            natoms,
            atoms,
            coords,
            q,
            ..Default::default()
        }
    }
}

pub fn mol_from_xyz_file(filename: &str) -> Result<XYZMolecule, Box<dyn Error>> {
    let contents = fs::read_to_string(filename)?;
    mol_from_xyz_string(&contents)
}

pub fn mol_from_xyz_string(contents: &str) -> Result<XYZMolecule, Box<dyn Error>> {
    let mol = parse_xyz_contents(&contents)?;
    Ok(mol)
}

pub fn molblock_from_xyz_string(contents: &str) -> Result<String, Box<dyn Error>> {
    let mol = parse_xyz_contents(&contents)?;
    let df = predict_mol(&mol);
    let molblock = create_molblock(mol, df)?;
    Ok(molblock)
}



/// Returns all files in directory with extension
pub fn scan_directory(path: &str, extension: &str) -> Vec<PathBuf> {
    let paths = fs::read_dir(path).unwrap();
    let mut path_vec = Vec::<PathBuf>::new();
    for path in paths {
        let p = path.unwrap().path();
        if p.extension().unwrap() == extension {
            path_vec.push(p);
        }
    }
    path_vec
}

///Parsing the contents of an xyz file
fn parse_xyz_contents(contents: &str) -> Result<XYZMolecule, Box<dyn Error>> {
    let lines = contents.split("\n");
    let mut atoms: Vec<String> = Vec::new();
    let mut coords: Vec<Float> = Vec::new();
    let mut nrows = 0;
    let mut natoms: usize = 0;
    let mut info: &str = "";

    for (i, line) in lines.enumerate() {
        if i == 0 {
            natoms = line.trim().parse()?;
        }
        if i == 1 {
            info = line;
        }
        if i > 1 && line.trim().len() > 0 {
            nrows += 1;
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
    assert_eq!(natoms as usize, nrows);
    let coords = Array2::from_shape_vec((nrows, 3), coords).unwrap();
    let mut molecule = XYZMolecule::new(atoms, coords, 0);
    molecule.info = info.to_owned();
    Ok(molecule)
}

/// Create a 2D ndarray with local bond information from distance matrix
/// https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html#similarities
pub fn create_dataframe(mol: &XYZMolecule) -> Result<DataFrame, Box<dyn Error>> {
    let dm = distance_matrix(&mol.coords);
    assert_eq!(mol.natoms, dm.ncols());
    let mut header = Vec::<String>::new();
    let mut features = Vec::<Vec<Float>>::new();
    //println!("first:{:?}",features);
    //iterate over rows of distance matrix
    println!();
    for i in 0..dm.ncols() {
        for j in 0..dm.ncols() {
            if i >= j {
                continue;
            }
            let dist = dm[[i, j]];
            if dist > DIST_CUTOFF {
                continue;
            }
            let mut i_tmp = i;
            let mut j_tmp = j;
            let e1 = &mol.atoms[i_tmp];
            let e2 = &mol.atoms[j_tmp];
            let mut an1 = ELEMENTS.iter().position(|&s| s == e1).unwrap_or_default();
            let mut an2 = ELEMENTS.iter().position(|&s| s == e2).unwrap_or_default();
            if an1 < an2 {
                mem::swap(&mut i_tmp, &mut j_tmp);
                mem::swap(&mut an1, &mut an2);
            }
            //assign to 2d array
            let mut data_row = vec![];
            data_row.push(i_tmp as Float + 1.0);
            data_row.push(j_tmp as Float + 1.0);
            data_row.push(mol.q as Float);
            data_row.push(an1 as Float);
            data_row.push(an2 as Float);
            data_row.push(dist);
            if features.len() == 0 {
                header.append(&mut vec![
                    "id1".to_string(),
                    "id2".to_string(),
                    "q".to_string(),
                    "ata".to_string(),
                    "atb".to_string(),
                    "distab".to_string(),
                ]);
            }
            //now go over neighbors of i an j
            for a in [i_tmp, j_tmp] {
                let b = if a == i_tmp { j_tmp } else { i_tmp };
                let label = if a == i_tmp { "a" } else { "b" };
                let label2 = if a == i_tmp { "b" } else { "a" };
                let row = dm.row(a).to_vec();
                let row_sorted = argsort(&row);
                let mut k = 0;
                for nextn in row_sorted.into_iter() {
                    if nextn == j_tmp || nextn == i_tmp {
                        continue;
                    }
                    if k >= N_CUT {
                        break;
                    };
                    let dist = dm[[a, nextn]];
                    let el_next = &mol.atoms[nextn];
                    let an_next = ELEMENTS
                        .iter()
                        .position(|&s| s == el_next)
                        .unwrap_or_default();
                    let distb = dm[[b, nextn]];

                    data_row.push(an_next as Float);
                    data_row.push(dist);
                    data_row.push(distb);
                    if features.len() == 0 {
                        let astr: String = format!("{}{}{}", "at", label, k + 1);
                        let bstr: String = format!("{}{}{}", "dist", label, k + 1);
                        let cstr: String = format!("{}{}{}{}", "dist", label, k + 1, label2);
                        header.append(&mut vec![astr, bstr, cstr]);
                    }
                    k += 1;
                }
            }
            features.push(data_row);
        }
    }
    let features_col = transpose(features);
    let mut series = Vec::<Series>::new();
    assert_eq!(features_col.len(), header.len());
    let zip_iter = features_col.iter().zip(header.iter());
    for (col, name) in zip_iter {
        let s = Series::new(name, col.to_owned());
        series.push(s);
    }
    let df = DataFrame::new(series).unwrap();
    Ok(df)
}

pub fn get_bonds(df: &DataFrame) -> u32 {
    let bonds = df.column("preds").unwrap();
    let mask = bonds.gt(0);
    let nbonds = mask.sum().unwrap();
    nbonds
}

pub fn create_molblock(mol: XYZMolecule, df: DataFrame) -> Result<String, Box<dyn Error>> {
    let bonds = df.column("preds")?;
    let mask = bonds.gt(0);
    let df = df.filter(&mask)?;

    let natoms = mol.atoms.len();
    let nbonds = mask.sum().unwrap();

    let mut ins: String = mol.name + "\n";

    // comment block
    ins += "ML generated sdf\n\n";
    ins += format!("{} {}  0  0  0  0  0  0  0  0  1 V2000\n", natoms, nbonds).as_str();

    // atom block
    for (at, xyz) in mol.atoms.iter().zip(mol.coords.outer_iter()) {
        ins += format!(
            "{:10.4}{:10.4}{:10.4} {:<2} 0  0  0  0  0\n",
            xyz[0], xyz[1], xyz[2], at
        )
        .as_str();
    }

    // bond block
    let (nrows, ncols) = df.shape();

    for i in 0..nrows {
        let row = df.get(i).unwrap();
        let id1 = &row[0].to_string();
        let id2 = &row[1].to_string();
        let bond = &row[ncols - 1].to_string();
        ins += format!("{:>3}{:>3}{:>3} 0  0  0  0  0\n", id1, id2, bond).as_str();
    }
    Ok(ins)
}

#[cfg(test)]
mod tests {
    use crate::ml::predict_mol;

    use super::*;
    #[test]
    fn test_molecule() {
        let atoms: Vec<String> = vec!["C".to_string(), "O".to_string()];
        let coords: Array2<Float> = arr2(&[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        let q: i32 = 0;
        let mol = XYZMolecule::new(atoms, coords, q);
        assert_eq!(mol.atoms.len(), 2);
    }
    #[test]
    fn parse_xyz_string() {
        let mol_str = "2

        C          0.00000        0.00000        0.00000
        O          0.00000        0.00000        1.00000";
        let mol = mol_from_xyz_string(mol_str).expect("Failed parsing!");
        assert_eq!(mol.coords.len(), 6);
        assert_eq!(mol.atoms.len(), 2);
        let df = predict_mol(&mol);
        println!("{}",df);
        let molblock = create_molblock(mol,df).expect("Failed molblock!");
        assert_eq!(molblock.len(), 176);
    }
    #[test]
    fn parse_xyz() {
        let mol = mol_from_xyz_file("data/test1.xyz").expect("Could not open file!");
        assert_eq!(mol.coords.len(), 69);
        assert_eq!(mol.atoms.len(), 23);
    }
    #[test]
    fn parse_all() {
        let entries = fs::read_dir("./data").unwrap();
        for (i, entry) in entries.enumerate() {
            let path = entry.unwrap().path();
            let ext = path.extension().unwrap().to_str().unwrap();
            if path.is_file() && ext == "xyz" {
                let fname = path.to_str().unwrap();
                let mol = mol_from_xyz_file(fname).expect("Could not read file!");
                distance_matrix(&mol.coords);
                let df = create_dataframe(&mol);
                assert!(df.is_ok());
            }
        }
    }
    #[test]
    fn test_df() {
        let mol = mol_from_xyz_file("data/test1.xyz").expect("Could not open file!");
        let df = create_dataframe(&mol).unwrap();
        println!("df.shape:{:?}", df.shape());
        assert_eq!(df.shape().0, 90);
        assert_eq!(df.shape().1, 24);
    }
    #[test]
    fn test_scandir() {
        let pvec = scan_directory("./data", "xyz");
        assert_eq!(pvec.len(), 6);
    }
    #[test]
    fn test_all() {
        let pvec = scan_directory("./data", "xyz");
        for (i, p) in pvec.iter().enumerate() {
            let contents = fs::read_to_string(p).expect("Could not open file!");
            let molblock = molblock_from_xyz_string(&contents);
            assert!(molblock.is_ok());
        }
    }
}
