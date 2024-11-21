use simd_test::{simd_dot_product_mixed_precision, simd_dot_product, simd_threaded_dot_product, dot_product};
use std::time::Instant;

fn main() {
    let n = 64_000_000;
    let vec_a = vec![0.000001; n];
    let vec_b = vec![0.000001; n];

    let start = Instant::now();
    let simd_result = simd_dot_product(&vec_a, &vec_b);
    let simd_dur = start.elapsed();

    let start = Instant::now();
    let seq_result = dot_product(&vec_a, &vec_b);
    let seq_dur = start.elapsed();

    let start = Instant::now();
    let simd_mixed_result = simd_dot_product_mixed_precision(&vec_a, &vec_b);
    let simd_mixed_dur = start.elapsed();

    let start = Instant::now();
    let simd_threaded_result = simd_threaded_dot_product(&vec_a, &vec_b);
    let simd_threaded_dur = start.elapsed();

    println!("SIMD:\nDot product: {}\nDuration: {:?}\n\n", simd_result, simd_dur);
    println!("Sequential:\nDot product: {}\nDuration: {:?}\n\n", seq_result, seq_dur);
    println!("SIMD Mixed Precision:\nDot product: {}\nDuration: {:?}\n\n", simd_mixed_result, simd_mixed_dur);
    println!("SIMD Threaded Precision:\nDot product: {}\nDuration: {:?}\n\n", simd_threaded_result, simd_threaded_dur);
}
