use crate::mamba::Float;

use ndarray::{arr1, arr2, indices_of, Array, Array2, ArrayView1, Axis};
use ndarray::{array, s};
use polars::chunked_array::ChunkedArray;
use polars::prelude::{DataFrame, ChunkAnyValue, Float32Type, AnyValue};

use std::any::Any;
use std::convert::TryInto;

//https://rust-lang-nursery.github.io/rust-cookbook/science/mathematics/linear_algebra.html

///Distance of 2 vectors
fn l2_dist(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    let diff = a - b;
    let res = (&diff * &diff).sum();
    return res.sqrt();
}

///convert vector into array representation
pub fn vec2array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

///convert dataframe data into flat vector
pub fn df2vec(df: &DataFrame) -> Vec<Float> {
    let (n_rows, n_cols) = df.shape();
    let mut flat_vec: Vec<Float> = Vec::<Float>::with_capacity(n_rows * n_cols);
    for idx in 0..n_rows {
        // get row at idx
        let row = df.get_columns().iter().map(|s| s.get(idx)).collect::<Vec<AnyValue>>();
        //transfrom row into f32
        let mut row2 = row.into_iter().map(|x| if let AnyValue::Float32(v) = x {v} else { 0.0 }).collect::<Vec<Float>>();
        flat_vec.append(&mut row2);
    }
    return flat_vec;
}

///Computing the distance matrix for a 2D array of coordinates.
pub fn distance_matrix(coords: &Array2<Float>) -> Array2<Float> {
    let mut distmat = Array2::zeros((coords.nrows(), coords.nrows()));
    let indices = indices_of(&distmat);
    for (i, j) in indices {
        let a = coords.row(i);
        let b = coords.slice(s![j, ..]); //equivalent
        let dist: Float = l2_dist(&a, &b);
        distmat[[i, j]] = dist;
    }
    distmat
}

/// Simple implementation of a argsort for an vector.  
/// Method returns the indices of a sorted vector.
/// Could probably much more efficient within ndarray.
pub fn argsort(v: &Vec<Float>) -> Vec<usize> {
    let mut vi: Vec<(usize, &Float)> = v.iter().enumerate().map(|(i, e)| (i, e)).collect();
    vi.sort_by(|&(_, a), &(_, b)| a.partial_cmp(b).unwrap());
    let sorted_index: Vec<usize> = vi.iter().map(|(i, _)| i.to_owned()).collect();
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

///computes the accuracy of two arrays
pub fn accuracy(preds: &[f32], y: &[f32]) -> f32 {
    let mut num_errors = 0;
    for (pred, label) in preds.iter().zip(y) {
        let pred_int: i32= *pred as i32 ;
        let label_int: i32 = *label as i32;
        if pred_int != label_int {
            num_errors += 1;
        }
    }
    let acc = 1.0 - (num_errors as f32 / preds.len() as f32);
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_acc() {
        let a = vec![1.0,2.0,5.0,4.0,3.0];
        let b = vec![1.0,2.0,2.0,4.0,3.0];
        let acc = accuracy(&a,&b);
        assert!(isclose(acc,0.8,1e-15));
    }
    #[test]
    fn test_distmat() {
        use ndarray_rand::rand_distr::Uniform;
        use ndarray_rand::RandomExt;
        let a = Array::random((20, 3), Uniform::new(0., 10.));
        println!("{:?}", &a);
        distance_matrix(&a);
        let s = &a.shape();
        assert_eq!(s[0] as i32, 20);
    }
    #[test]
    fn test_argsort() {
        let v: Vec<Float> = vec![-5.0, 4.9, 1.1, -3.1, 2.1];
        let vi = argsort(&v);
        println!("{:?}", vi);
        assert_eq!(vi, vec![0, 3, 2, 4, 1]);
    }
    #[test]
    fn test_l2dist() {
        let a = array![0., 1., 0.];
        let a = a.view();
        let b = array![1., 0., 1.];
        let b = b.view();
        let dist = l2_dist(&a, &b);
        assert!(isclose(dist, Float::sqrt(3.0), 1e-15));
    }
    #[test]
    fn test_vec2array() {
        let v: Vec<i32> = vec![0,1,2];
        let a: [i32;  3]  = vec2array(v);
        assert_eq!(a, [0, 1, 2]);
    }
    #[test]
    fn test_isclose() {
        assert!(isclose(1.414, Float::sqrt(2.0), 1e-3));
    }
}
