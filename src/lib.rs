#![feature(portable_simd)]
use std::simd::f32x8;
use std::simd::num::SimdFloat;
const NUM_LANES: usize = 8;

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
    while i < a.len() {
        sum += a[i] * b[i];
        i += 1;
    }

    sum
}

pub fn simd_dot_product_mixed_precision(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());

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

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());

    a.iter()
        .zip(b.iter())        
        .map(|(x, y)| x * y)  
        .sum()               
}