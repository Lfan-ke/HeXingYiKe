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
                    data[offset + j] = e; // Q：所以这里改变了原有内存的值啊啊啊？
                    e
                })
                .sum::<f32>();

            (0..boundary).for_each(|j| data[offset + j] /= sum);
            (boundary..total_seq_len).for_each(|j| data[offset + j] = 0.0);
        }
    }
}

pub fn rms_norm(y: &mut Tensor<f32>, x: &Tensor<f32>, w: &Tensor<f32>, epsilon: f32) {
    let ndim = y.shape().len();
    assert!(ndim >= 2);
    let seq_len = y.shape()[ndim - 2]; // xi yi
    let total_seq_len = y.shape()[ndim - 1]; // n
    let batch = y.size() / (seq_len * total_seq_len);
    assert_eq!(w.shape()[w.shape().len() - 1], total_seq_len);
    let (_x, _y, _w) = (x.data(), unsafe { y.data_mut() }, w.data());
    for b in 0..batch {
        let base = b * seq_len * total_seq_len;
        for i in 0..seq_len {
            // xi yi
            let offset = base + i * total_seq_len;
            let xi2_mean = (0..total_seq_len)
                .map(|idx| _x[offset + idx] * _x[offset + idx])
                .sum::<f32>()
                / total_seq_len as f32;

            let rms = (xi2_mean + epsilon).sqrt();
            for j in 0..total_seq_len {
                _y[offset + j] = (_w[j] * _x[offset + j]) / rms;
            }
        }
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    sigmoid(x) * x
}

// y = silu(x) * y
// hint: this is an element-wise operation
pub fn swiglu(y: &mut Tensor<f32>, x: &Tensor<f32>) {
    let len = y.size();
    assert!(len == x.size());

    let _y = unsafe { y.data_mut() };
    let _x = x.data();

    for i in 0..len {
        _y[i] *= silu(_x[i]);
    }
}

// C = beta * C + alpha * A @ B^T
// hint: You don't need to do an explicit transpose of B
pub fn matmul_transb(c: &mut Tensor<f32>, beta: f32, a: &Tensor<f32>, b: &Tensor<f32>, alpha: f32) {
    /// 默认第二个向量制动专职，即：b，是b.T！
    /// `A` 形状为 `m×k`，`B` 形状为 `n×k`，`C` 形状为 `m×n`
    // 目前偷个小懒，暂不实现广播
    let c_size = c.size().clone();
    let c_shape = c.shape().clone();
    let (a_dim, b_dim, c_dim) = (a.shape().len(), b.shape().len(), c_shape.len());
    // 你可以默认输入输出都是二维矩阵，即 `A` 形状为 `m×k`，`B` 形状为 `n×k`，`C` 形状为 `m×n`，可以不用考虑广播的情况
    let (_a, _b, _c) = (a.data(), b.data(), unsafe { c.data_mut() });
    assert!(a_dim >= 2);
    assert!(b_dim >= 2);
    assert!(c_dim >= 2);
    assert!(a.shape()[a_dim - 2] == c_shape[c_dim - 2]); // m
    assert!(a.shape()[a_dim - 1] == b.shape()[b_dim - 1]); // k
    assert!(b.shape()[b_dim - 2] == c_shape[c_dim - 1]); // n
    let (m, k, n) = (
        a.shape()[a_dim - 2],
        a.shape()[a_dim - 1],
        b.shape()[b_dim - 2],
    );
    // 以C为基准进行循环最好
    let batch = c_size / (m * n);
    // 在有`batch`的情况下对abc进行索引
    let clip_a = |bh, i, j| {
        let base = bh * m * k;
        _a[base + i * k + j]
    };
    let clip_b = |bh, i, j| {
        let base = bh * k * n;
        _b[base + i * k + j]
    };
    for bh in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let base = bh * m * n;
                _c[base + mi * n + ni] *= beta;
                _c[base + mi * n + ni] += alpha
                    * (0..k)
                        .map(|ki| clip_a(bh, mi, ki) * clip_b(bh, ni, ki))
                        .sum::<f32>();
            }
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
