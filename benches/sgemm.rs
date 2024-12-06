use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rust_cuda_demo::cublas;
use std::hint::black_box;

pub fn cpu_sgemm(n: usize, a: &[f32], b: &[f32], alpha: f32, beta: f32) -> Vec<f32> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut prod = 0.0;
            for k in 0..n {
                prod += a[k * n + i] * b[j * n + k];
            }
            c[j * n + i] = alpha * prod + beta * c[j * n + i];
        }
    }
    c
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu");
    group.sample_size(10);
    for n in [10, 100, 1000, 2_000].iter() {
        let a_matrix: Vec<_> = (0..(n * n)).map(|x| x as f32).collect();
        let b_matrix: Vec<_> = (0..(n * n)).map(|x| x as f32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                let c_matrix = cpu_sgemm(n, black_box(&a_matrix), black_box(&b_matrix), 1.0, 0.0);
                assert_eq!(c_matrix.len(), n * n);
            })
        });
    }
    group.finish();

    let mut group = c.benchmark_group("gpu");
    group.sample_size(10);
    for n in [10, 100, 1000, 2_000].iter() {
        let a_matrix: Vec<_> = (0..(n * n)).map(|x| x as f32).collect();
        let b_matrix: Vec<_> = (0..(n * n)).map(|x| x as f32).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |b, &n| {
            b.iter(|| {
                let c_matrix =
                    cublas::sgemm(n, black_box(&a_matrix), black_box(&b_matrix), 1.0, 0.0).unwrap();
                assert_eq!(c_matrix.len(), n * n);
            })
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
