use crate::tensor::Tensor;

// get (row) vectors from a 2D table given a list of indices
pub fn gather(y: &mut Tensor<f32>, indices: &Tensor<u32>, table: &Tensor<f32>) {
    let length = indices.size();
    let table_shape = table.shape();
    assert!(table_shape.len() == 2);
    let dim = table_shape[1];
    assert!(y.size() == length * dim);
    for i in 0..length {
        let src = &table.data()[indices.data()[i] as usize * dim..][..dim];
        let dst = &mut unsafe { y.data_mut() }[i * dim..][..dim];
        dst.copy_from_slice(src);
    }
}

// RoPE: Rotary Positional Embedding
pub fn rope(y: &mut Tensor<f32>, start_pos: usize, theta: f32) {
    let shape = y.shape();
    assert!(shape.len() == 3);
    let seq_len = shape[0];
    let n_heads = shape[1];
    let d = shape[2];
    let data = unsafe { y.data_mut() };
    for tok in 0..seq_len {
        let pos = start_pos + tok;
        for head in 0..n_heads {
            for i in 0..d / 2 {
                let a = data[tok * n_heads * d + head * d + i];
                let b = data[tok * n_heads * d + head * d + i + d / 2];
                let freq = pos as f32 / theta.powf((i * 2) as f32 / d as f32);
                let (sin, cos) = freq.sin_cos();
                data[tok * n_heads * d + head * d + i] = a * cos - b * sin;
                data[tok * n_heads * d + head * d + i + d / 2] = b * cos + a * sin;
            }
        }
    }
}

// softmax(x) = exp(x - max) / sum(exp(x - max))
// y = softmax(mask(x))
pub fn masked_softmax(y: &mut Tensor<f32>) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2];
    let total_seq_len = y.shape()[ndim - 1];
    let batch = y.size() / (seq_len * total_seq_len);
    let data = unsafe { y.data_mut() };
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            let offset = base + i * total_seq_len;
            let boundary = total_seq_len - seq_len + i + 1;

            let max = data[offset..offset + boundary]
                .iter()
                .fold(data[offset], |a, b| a.max(*b));

            let sum = (0..boundary)
                .map(|j| {
                    let e = (data[offset + j] - max).exp();
                    data[offset + j] = e;
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let shape = y.shape().clone();
    let len = shape.len();
    let last_dim = len - 1;
    let _y = unsafe { y.data_mut() };
    let _x = x.data();
    let _w = w.data();

    let mut ext_loop = 1;
    for i in 0..(shape.len() - 1) {
        ext_loop *= shape[i];
    }
    let inner_size = shape[last_dim];

    for i in 0..ext_loop {
        let mut xp = 0f32;
        for j in 0..shape[last_dim] {
            xp += _x[i * inner_size + j] * _x[i * inner_size + j];
            _y[i * inner_size + j] = _w[j] * _x[i * inner_size + j];
        }
        xp = f32::sqrt(xp / inner_size as f32 + epsilon);
        for j in 0..shape[last_dim] {
            _y[i * inner_size + j] /= xp;
        }   
    }
}
// y = sigmoid(x) * x * y
// hint: this is an element-wise operation
pub fn silu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        _y[i] = _y[i] * _x[i] / (1f32 + f32::exp(-_x[i]));
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    let shape_c = c.shape();
    let (c_rows, c_cols) = (shape_c[0], shape_c[1]);
    let inner = a.shape()[1];
    let _c = unsafe { c.data_mut() };
    let _a = a.data();
    let _b = b.data();

    // Scale c by beta
    for val in _c.iter_mut() {
        *val *= beta;
    }

    // Perform matrix multiplication
    for x in 0..c_rows {
        for y in 0..c_cols {
            let mut sum = 0f32;
            for k in 0..inner {
                sum += _a[x * inner + k] * _b[y * inner + k];
            }
            _c[x * c_cols + y] += alpha * sum;
        }
    }
}
// Dot product of two tensors (treated as vectors)
#[allow(unused)]
pub fn dot(x: &Tensor<f32>, y: &Tensor<f32>) -> f32 {
    let len = x.size();
    assert!(len == y.size());
    let x_ = x.data();
    let y_ = y.data();
    let mut sum = 0.0;
    for i in 0..len {
        sum += x_[i] * y_[i];
    }
    sum
}

// Sample a index from a tensor (treated as a probability vector)
pub fn random_sample(x: &Tensor<f32>, top_p: f32, top_k: u32, temperature: f32) -> u32 {
    assert!(x.shape()[x.shape().len() - 1] == x.size());
    if temperature <= 0. || top_k < 2 || top_p <= 0. {
        return x
            .data()
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as _;
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    struct Probability {
        val: f32,
        tok: u32,
    }
    impl Eq for Probability {}
    impl PartialOrd for Probability {
        #[inline]
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for Probability {
        #[inline]
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            match self.val.total_cmp(&other.val) {
                std::cmp::Ordering::Equal => self.tok.cmp(&other.tok),
                ord => ord.reverse(),
            }
        }
    }
    impl From<(usize, &f32)> for Probability {
        #[inline]
        fn from((i, p): (usize, &f32)) -> Self {
            Self {
                val: p.clone(),
                tok: i as _,
            }
        }
    }

    // sort
    let mut logits = x
        .data()
        .iter()
        .enumerate()
        .map(Probability::from)
        .collect::<Vec<_>>();
    logits.sort_unstable();
    let max = core::mem::replace(&mut logits[0].val, 1.);
    // softmax & sum
    for i in 1..logits.len() {
        logits[i].val = logits[i - 1].val + ((logits[i].val - max) / temperature).exp();
    }
    // topk & topp & random
    let pk = logits[(top_k as usize).min(logits.len()) - 1].val;
    let pp = logits[logits.len() - 1].val * top_p;
    let plimit = rand::random::<f32>() * f32::min(pk, pp);
    // sample
    logits.iter().find(|p| p.val >= plimit).unwrap().tok
}

// Your implementation should at least pass the following tests:
#[test]
fn test_silu() {
    let mut y = Tensor::<f32>::new(vec![2., 3., 4.], &vec![1, 3]);
    let x = Tensor::<f32>::new(vec![1., 2., 3.], &vec![1, 3]);
    silu(&mut y, &x);
    assert!(y.close_to(
        &Tensor::<f32>::new(vec![1.4621172, 5.2847824, 11.43089], &vec![1, 3]),
        1e-3
    ));
}

#[test]
fn test_rms_norm() {
    let mut y = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let x = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let w = Tensor::<f32>::new(vec![1., 2.], &vec![2]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![0.6324554, 2.5298216, 0.8485281, 2.2627416],
            &vec![2, 2]
        ),
        1e-3
    ));
}

#[test]
fn test_rms_norm_0() {
    let mut y = Tensor::<f32>::new( vec![
        1.3047, -1.9949, 0.5328,
        0.7988, -0.0593, 0.3121,
    ], &vec![2, 3]);
    let x = Tensor::<f32>::new( vec![
        1.3047, -1.9949, 0.5328,
        0.7988, -0.0593, 0.3121,
    ], &vec![2, 3]);

    let w = Tensor::<f32>::new(vec![1., 2., 3.], &vec![3]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![
                0.9252, -2.8293, 1.1336,
                1.6095, -0.2390, 1.8862,
            ],
            &vec![2, 3]
        ),
        1e-3
    ));
}

#[test]
fn test_rms_norm_1() {
    let mut y = Tensor::<f32>::new(vec![
        -0.0338, -1.4320, -1.4298, -0.0493,
        0.4963, 2.3341, 0.6086, 0.1502,
        1.2347, -0.1080, -0.6381, -1.2577,
        0.3982, 0.6274, 0.6667, -0.3212,
        1.4439, -0.4832, 0.5520, 0.5102,
        -1.1528, -1.3846, -2.4974, 1.3092,
        -0.9207, -0.9543, -0.0921, -0.8487,
    ], &vec![7, 4]);
    let x = Tensor::<f32>::new(vec![
        -0.0338, -1.4320, -1.4298, -0.0493,
        0.4963, 2.3341, 0.6086, 0.1502,
        1.2347, -0.1080, -0.6381, -1.2577,
        0.3982, 0.6274, 0.6667, -0.3212,
        1.4439, -0.4832, 0.5520, 0.5102,
        -1.1528, -1.3846, -2.4974, 1.3092,
        -0.9207, -0.9543, -0.0921, -0.8487,
    ], &vec![7, 4]);

    let w = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![4]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![
                -0.0334, -2.8294, -4.2375, -0.1949,
                0.4023, 3.7842, 1.4800, 0.4870,
                1.3153, -0.2301, -2.0390, -5.3590,
                0.7594, 2.3930, 3.8144, -2.4499,
                1.7006, -1.1383, 1.9505, 2.4039,
                -0.6890, -1.6551, -4.4780, 3.1301,
                -1.1675, -2.4206, -0.3503, -4.3052,
            ],
            &vec![7, 4]
        ),
        1e-3
    ));
}

#[test]
fn test_rms_norm_2() {
    let mut y = Tensor::<f32>::new(vec![
            -2.1117, 0.2419, 0.3274, 
            -2.6815, -0.5004, -0.5681,
            0.6042, -0.3003, -0.0382,
            -1.5544, -1.0339, -0.3826
        ], &vec![2, 2, 3]);
    let x = Tensor::<f32>::new(vec![
        -2.1117, 0.2419, 0.3274, 
        -2.6815, -0.5004, -0.5681,
        0.6042, -0.3003, -0.0382,
        -1.5544, -1.0339, -0.3826
    ], &vec![2, 2, 3]);

    let w = Tensor::<f32>::new(vec![1., 2., 3.], &vec![3]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![
                -1.7007, 0.3896, 0.7911,
                -1.6669, -0.6221, -1.0594,
                1.5486, -1.5393, -0.2937,
                -1.4128, -1.8794, -1.0432,
            ],
            &vec![2, 2, 3]
        ),
        1e-3
    ));
}

#[test]
fn test_rms_norm_3() {
    let mut y = Tensor::<f32>::new(vec![
        -0.6739, 0.3571, -0.8788,
        -1.2909, 1.1852, -1.2400,
        0.5001, -0.3792, 0.5342,
        1.0144, 0.8592, -0.2071,
        0.3115, -0.2703, 0.9758,
        -0.7850, 0.4510, -1.5042,
        0.5577, -0.5024, 0.5586,
        0.1071, 0.4731, -2.4975,
        1.2519, -0.8391, 0.1562,
        1.7349, 0.2805, -0.1348,
        -1.3780, -1.0139, -0.3333,
        -0.1251, 0.8297, 0.4957,
    ], &vec![2, 2, 3, 3]);
    let x = Tensor::<f32>::new(vec![
        -0.6739, 0.3571, -0.8788,
        -1.2909, 1.1852, -1.2400,
        0.5001, -0.3792, 0.5342,
        1.0144, 0.8592, -0.2071,
        0.3115, -0.2703, 0.9758,
        -0.7850, 0.4510, -1.5042,
        0.5577, -0.5024, 0.5586,
        0.1071, 0.4731, -2.4975,
        1.2519, -0.8391, 0.1562,
        1.7349, 0.2805, -0.1348,
        -1.3780, -1.0139, -0.3333,
        -0.1251, 0.8297, 0.4957,
    ], &vec![2, 2, 3, 3]);

    let w = Tensor::<f32>::new(vec![10., 12., 8.], &vec![3]);
    rms_norm(&mut y, &x, &w, 1e-6);
    assert!(y.close_to(
        &Tensor::<f32>::new(
            vec![
                -10.0316, 6.3787, -10.4647,
                -10.4149, 11.4752, -8.0034,
                10.5095, -9.5639, 8.9810,
                13.0592, 13.2734, -2.1326,
                5.0921, -5.3039, 12.7633,
                -7.7445, 5.3397, -11.8719,
                10.3242, -11.1602, 8.2721,
                0.7293, 3.8650, -13.6022,
                14.3111, -11.5102, 1.4286,
                17.0484, 3.3072, -1.0596,
                -13.6936, -12.0906, -2.6495,
                -2.2236, 17.6953, 7.0476,
            ],
            &vec![2, 2, 3, 3]
        ),
        1e-3
    ));
}

#[test]
fn test_matmul_transb() {
    let mut c = Tensor::<f32>::new(vec![1., 2., 3., 4.], &vec![2, 2]);
    let a = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![1., 2., 3., 4., 5., 6.], &vec![2, 3]);
    matmul_transb(&mut c, 1., &a, &b, 1.);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![15., 34., 35., 81.], &vec![2, 2]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_0() {
    let mut c = Tensor::<f32>::new(vec![
        -0.8113, 1.8005, 1.4450, 0.5919,
        0.4683, -1.2566, -1.1469, 0.2845,
    ], &vec![2, 4]);
    let a = Tensor::<f32>::new(vec![
        -0.7777, -0.8577, -0.3442,
        1.1499, 0.6590, 0.5645,
    ], &vec![2, 3]);
    let b = Tensor::<f32>::new(vec![
        -0.1603, 0.1595, 1.0277,
        -0.3535, -0.4132, 0.6107,
        -0.9211, 0.0353, -0.2416,
        -1.4986, -0.4596, -0.1550,
    ], &vec![3, 4]);
    matmul_transb(&mut c, 2.43, &a, &b, 1.41);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![
            -2.4874, 4.9662, 4.5959, 3.7126,
            1.8443, -3.5247, -4.4398, -2.2889,
        ], &vec![2, 4]),
        1e-3
    ));
}

#[test]
fn test_matmul_transb_1() {
    let mut c = Tensor::<f32>::new(vec![
        0.5959, -1.8501, -0.6161, 1.3742, -0.9042,
        -0.6141, -1.2398, 1.2380, -1.1101, 1.2153,
        -0.7181, 0.2656, -0.2232, 1.4864, 0.4870,
    ], &vec![3, 5]);
    let a = Tensor::<f32>::new(vec![
        0.6937, -0.7814, -0.5549, -0.4525, -1.2635,
        2.0139, -1.2418, -0.9660, -0.0260, -0.5243,
        0.6419, 0.4324, -0.8068, -1.3037, 0.0705,
    ], &vec![3, 5]);
    let b = Tensor::<f32>::new(vec![
        1.2066, -0.20014, -1.0151, -0.96161, -0.51317,
        0.19607, 0.73423, -2.6455, -0.08325, -0.21335,
        -0.51207, -1.6300, 0.61885, -0.92605, 1.1062,
        0.85047, -1.0426, 0.51591, 1.3225, 0.0024804,
        1.7679, -0.77657, -1.3297, 0.64967, 1.5565,
    ], &vec![5, 5]);
    matmul_transb(&mut c, 0.2, &a, &b, 0.2);
    assert!(c.close_to(
        &Tensor::<f32>::new(vec![
            0.6472, -0.1026, -0.2039, 0.3782, -0.1188,
            0.6678, 0.1826, 0.2154, 0.2727, 1.2383,
            0.4012, 0.5874, -0.0942, -0.1118, 0.3243,
        ], &vec![3, 5]),
        1e-3
    ));
}