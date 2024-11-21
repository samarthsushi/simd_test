#![feature(test)]
#![feature(portable_simd)]

extern crate test;

use std::simd::f32x8;
use std::simd::num::SimdFloat;
use rand::Rng;
use rayon::prelude::*;

const NUM_LANES: usize = 8;

pub fn generate_random_vector(n: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n).map(|_| rng.gen_range(0.0..1.0)).collect()
}

pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    let mut sum_simd = f32x8::splat(0.0);
    let mut i = 0;

    while i + NUM_LANES <= a.len() {
        let a_chunk = f32x8::from_slice(&a[i..i + NUM_LANES]);
        let b_chunk = f32x8::from_slice(&b[i..i + NUM_LANES]);
        sum_simd += a_chunk * b_chunk;
        i += NUM_LANES;
    }

    let mut sum = sum_simd.reduce_sum();

    // Handle the remaining elements that didn't fit into a SIMD chunk
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

pub fn simd_dot_product_mixed_precision(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());

    // Accumulate results in an f64 vector
    let mut sum_simd = [0.0; NUM_LANES];

    let mut i = 0;
    while i + NUM_LANES <= a.len() {
        let a_chunk = f32x8::from_slice(&a[i..i + NUM_LANES]);
        let b_chunk = f32x8::from_slice(&b[i..i + NUM_LANES]);
        let product = a_chunk * b_chunk;

        for (j, &prod) in product.to_array().iter().enumerate() {
            sum_simd[j] += prod as f64;
        }

        i += NUM_LANES;
    }
    let mut sum: f64 = sum_simd.iter().sum();

    while i < a.len() {
        sum += a[i] as f64 * b[i] as f64;
        i += 1;
    }

    sum
}

pub fn simd_threaded_dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    // Handle the full SIMD chunks
    let simd_sum: f32 = a.par_chunks_exact(NUM_LANES)
        .zip(b.par_chunks_exact(NUM_LANES))
        .map(|(a_chunk, b_chunk)| {
            let a_simd = f32x8::from_slice(a_chunk);
            let b_simd = f32x8::from_slice(b_chunk);
            let product = a_simd * b_simd;

            // Sum the SIMD results
            product.reduce_sum()
        })
        .sum();

    // Handle the remaining elements that don't fit into a full SIMD chunk
    let remainder_sum: f32 = a.par_chunks_exact(NUM_LANES)
        .remainder()
        .iter()
        .zip(b.par_chunks_exact(NUM_LANES).remainder().iter())
        .map(|(&x, &y)| x * y)
        .sum();

    // Combine the results
    simd_sum + remainder_sum
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())        
        .map(|(x, y)| x * y)  
        .sum()               
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;
    use std::hint::black_box;

    #[bench]
    fn bench_simd(b: &mut Bencher) {
        let n = 64_000_000;
        let vec_a = vec![0.000001; n];
        let vec_b = vec![0.000001; n];
        b.iter(|| simd_dot_product(black_box(&vec_a), black_box(&vec_b)));
    }

    #[bench]
    fn bench_sequential(b: &mut Bencher) {
        let n = 64_000_000;
        let vec_a = vec![0.000001; n];
        let vec_b = vec![0.000001; n];
        b.iter(|| dot_product(black_box(&vec_a), black_box(&vec_b)));
    }

    #[bench]
    fn bench_simd_mixed_precision(b: &mut Bencher) {
        let n = 64_000_000;
        let vec_a = vec![0.000001; n];
        let vec_b = vec![0.000001; n];
        b.iter(|| simd_dot_product_mixed_precision(black_box(&vec_a), black_box(&vec_b)));
    }

    #[bench]
    fn bench_simd_threaded(b: &mut Bencher) {
        let n = 64_000_000;
        let vec_a = vec![0.000001; n];
        let vec_b = vec![0.000001; n];
        b.iter(|| simd_threaded_dot_product(black_box(&vec_a), black_box(&vec_b)));
    }
}

