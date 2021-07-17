
use std::fs;

use ndarray::{Axis,Array,Array2,arr1,arr2,ArrayView1,indices_of};
use ndarray::{s,array};

use crate::mamba;
use mamba::{Molecule,create_dataframe,mol_from_xyz};


pub type Float = f32;


//https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html

///Distance of 2 vectors
fn l2_dist(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    let diff = a - b;
    let res = (&diff * &diff).sum();
    return res.sqrt()
}

///Computing the distance matrix for a 2D array of coordinates.
pub fn distance_matrix(coords: &Array2<Float>) -> Array2<Float> {
    let mut distmat = Array2::zeros((coords.nrows(), coords.nrows()));
    let indices = indices_of(&distmat);
    for (i,j) in indices {
        //a.slice(s![1, .., ..])
        //let a = coords.slice(s![i, ..]);
        let a = coords.row(i);
        let b = coords.slice(s![j, ..]); //equivalent
        let dist : Float = l2_dist(&a,&b);
        distmat[[i, j]] = dist;
    }
    //println!("distmat:{} shape: {:?}",distmat,distmat.shape());
    distmat
}

/// Simple implementation of a argsort for an vector.  
/// Method returns the indices of a sorted vector.
/// Could probably much more efficient within ndarray.
pub fn argsort(v: &Vec<Float>) ->Vec<usize> {
    let mut vi : Vec<(usize, &Float)> = v.iter().enumerate().map(|(i,e)| (i,e)).collect();
    vi.sort_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap());
    let sorted_index: Vec<usize> = vi.iter().map(|(i,_)| i.to_owned()).collect();
    sorted_index
}

///Simple test for floating point equality
fn isclose(a: Float, b: Float, epsilon: Float) -> bool {
    (a - b).abs() <= a.abs().max(b.abs()) * epsilon
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_distmat() {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;
        let a = Array::random((20, 3), Uniform::new(0., 10.));
        distance_matrix(&a);
    }
    #[test]
    fn test_argsort() {
        let v: Vec<Float> = vec![-5.0, 4.9, 1.1, -3.1, 2.1];
        let vi = argsort(&v);
        println!("{:?}",vi);
        assert_eq!(vi,vec![0,3,2,4,1]);
    }
    #[test]
    fn test_l2dist() {
        let a = array![0.,1.,0.];
        let a = a.view();
        let b = array![1.,0.,1.];
        let b = b.view();
        let dist = l2_dist(&a, &b);
        assert!(isclose(dist,Float::sqrt(3.0),1e-15));
    }

    
 

}