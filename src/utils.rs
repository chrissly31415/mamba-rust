use crate::mamba::{Float};

use ndarray::{Axis,Array,Array2,arr1,arr2,ArrayView1,indices_of};
use ndarray::{s,array};


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

/// Transpode 2d vec: https://stackoverflow.com/questions/64498617/how-to-transpose-a-vector-of-vectors-in-rust
pub fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
    assert!(!v.is_empty());
    let len = v[0].len();
    //vector of inner iterators
    let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
    (0..len)
        .map(|_| {
            iters
                .iter_mut()
                .map(|n| n.next().unwrap())
                .collect::<Vec<T>>()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_distmat() {
        use ndarray_rand::RandomExt;
        use ndarray_rand::rand_distr::Uniform;
        let a = Array::random((20, 3), Uniform::new(0., 10.));
        println!("{:?}",&a);
        distance_matrix(&a);
        let s = &a.shape();
        assert_eq!(s[0] as i32,20);
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