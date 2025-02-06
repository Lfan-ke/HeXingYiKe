use std::cmp::{min, Ordering};
use std::fs::File;
use std::ops::DerefMut;
use std::vec;

use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::operators::{
    gather, masked_softmax, matmul_transb, random_sample, rms_norm, rope, swiglu,
};
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
use std::path::Path;
use std::sync::Arc;

pub struct Llama<T> {
    pub vocab: usize,           // vocab size
    pub n_layers: usize,        // number of layers
    pub n_q_h: usize, // number of heads for q  // Q头数量是KV头的整数倍，所以nqh = nkvh * n_groups
    pub n_kv_h: usize, // number of heads for k and v
    pub d: usize,     // dimension of hidden states
    pub dqkv: usize,  // length of a single q, k, or v vector
    pub di: usize,    // dimension of intermediate states
    pub eps: f32,     // epsilon for RMS normalization
    pub rope_theta: f32, // rope theta for rope initialization
    pub max_seq_len: usize, // maximum sequence length
    pub params: LLamaParams<T>, // trained weights of this model
    pub bos_token_id: u32, // start token id
    pub eos_token_id: u32, // end token id
}

fn easy_softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = *logits
        .iter()
        .reduce(|a, b| if a > b { a } else { b })
        .unwrap();
    let exp_logits: Vec<_> = logits.iter().map(|&x| (x - max_logit).exp()).collect();
    let sum_exp_logits: f32 = exp_logits.iter().sum();
    let result: Vec<f32> = exp_logits.iter().map(|&x| x / sum_exp_logits).collect();
    result.to_vec()
}

#[macro_export]
macro_rules! flush_print {
    ($fmt:expr) => ({
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt).unwrap();
        stdout.flush().unwrap();
    });
    ($fmt:expr, $($arg:tt)*) => ({
        use std::io;
        use std::io::Write;
        let mut stdout = io::stdout();
        write!(stdout, $fmt, $($arg)*).unwrap();
        stdout.flush().unwrap();
    });
}

/// 分别表示：当前推理出的词、是否最大、是否EOS、是否出错（比如开始时KVC已经达到最大了）
#[derive(Copy, Clone)]
pub struct GenResult(pub u32, pub bool, pub bool, pub bool);

impl GenResult {
    pub fn take(&self) -> u32 {
        self.0
    }

    pub fn is_full(&self) -> bool {
        self.1
    }

    pub fn is_eos(&self) -> bool {
        self.2
    }

    pub fn is_err(&self) -> bool {
        self.3
    }

    pub fn decode(&self) -> String {
        if self.is_err() || self.is_eos() {
            panic!("decode error, its err or eos!")
        }
        crate::options::deserialization(&[self.0])
    }
}

impl Llama<f32> {
    pub fn from_safetensors(model_dir: impl AsRef<Path>) -> Self {
        let config = File::open(model_dir.as_ref().join("config.json")).unwrap();
        let config: LlamaConfigJson = serde_json::from_reader(config).unwrap();
        let model_file = std::fs::read(model_dir.as_ref().join("model.safetensors")).unwrap();
        let safetensor = SafeTensors::deserialize(&model_file).unwrap();

        let params = LLamaParams::from_safetensors(&safetensor, &config);

        Self {
            vocab: config.vocab_size,
            n_layers: config.num_hidden_layers,
            n_q_h: config.num_attention_heads,
            n_kv_h: config.num_key_value_heads,
            d: config.hidden_size,
            dqkv: config.hidden_size / config.num_attention_heads,
            di: config.intermediate_size,
            eps: config.rms_norm_eps,
            rope_theta: config.rope_theta,
            max_seq_len: config.max_position_embeddings,
            params,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        }
    }

    pub fn new_cache(&self) -> KVCache<f32> {
        KVCache::new(self.n_layers, self.max_seq_len, self.n_kv_h * self.dqkv, 0)
    }

    pub fn forward(&self, input: &Tensor<u32>, cache: &mut KVCache<f32>) -> Tensor<f32> {
        let seq_len = input.size();
        let past_seq_len = cache.len();
        cache.increment(seq_len);
        let total_seq_len = past_seq_len + seq_len;
        let n_groups = self.n_q_h / self.n_kv_h;

        let mut residual = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut hidden_states = Tensor::<f32>::default(&vec![seq_len, self.d]);
        let mut q_buf = Tensor::<f32>::default(&vec![seq_len, self.n_q_h * self.dqkv]);
        let mut att_scores =
            Tensor::<f32>::default(&vec![self.n_kv_h, n_groups, seq_len, total_seq_len]);
        let mut gate_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);
        let mut up_buf = Tensor::<f32>::default(&vec![seq_len, self.di]);

        gather(&mut residual, input, &self.params.embedding_table);
        for layer in 0..self.n_layers {
            rms_norm(
                &mut hidden_states,
                &residual,
                &self.params.rms_att_w[layer],
                self.eps,
            );

            let q = (&mut q_buf).reshape(&vec![seq_len, self.n_q_h * self.dqkv]); // (seq, n_h * dqkv)
            let k = &mut cache.k_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            let v = &mut cache.v_cache(layer, past_seq_len); // (seq, n_kv_h * dqkv)
            matmul_transb(q, 0., &hidden_states, &self.params.wq[layer], 1.0);
            matmul_transb(k, 0., &hidden_states, &self.params.wk[layer], 1.0);
            matmul_transb(v, 0., &hidden_states, &self.params.wv[layer], 1.0);
            rope(
                q.reshape(&vec![seq_len, self.n_q_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );
            rope(
                k.reshape(&vec![seq_len, self.n_kv_h, self.dqkv]),
                past_seq_len,
                self.rope_theta,
            );

            let full_k = &mut cache.k_cache(layer, 0); // (total_seq, n_kv_h * dqkv)
            let full_v = &mut cache.v_cache(layer, 0); // (total_seq, n_kv_h * dqkv)

            self_attention(
                &mut hidden_states,
                &mut att_scores,
                q,
                full_k,
                full_v,
                self.n_kv_h,
                n_groups,
                seq_len,
                total_seq_len,
                self.dqkv,
            );

            matmul_transb(
                &mut residual,
                1f32,
                &hidden_states,
                &self.params.wo[layer],
                1f32,
            );

            mlp(
                &mut residual,
                &mut hidden_states,
                &mut gate_buf,
                &mut up_buf,
                &self.params.w_up[layer],
                &self.params.w_down[layer],
                &self.params.w_gate[layer],
                &self.params.rms_ffn_w[layer],
                self.eps,
            );
        }

        let mut logits = Tensor::<f32>::default(&vec![1, self.vocab]);
        let mut hidden_states = hidden_states.slice((seq_len - 1) * self.d, &vec![1, self.d]);
        let residual = residual.slice((seq_len - 1) * self.d, &vec![self.d]);

        rms_norm(
            &mut hidden_states,
            &residual,
            &self.params.rms_out_w,
            self.eps,
        );

        matmul_transb(&mut logits, 0., &hidden_states, &self.params.lm_head, 1.0);

        logits
    }

    pub fn generate(
        &self,
        token_ids: &[u32], // 从始至终传递全部的 ids ！
        max_len: usize,    // 模型允许的最大序列长度
        max_gen: usize,    // 最大允许生成长度
        top_p: f32,
        top_k: u32,
        temperature: f32,
        history: Arc<crate::options::History>,
    ) -> GenResult {
        /*  设计为yield结构：
         *      即：每次根据KVC检测应该推理单个token还是一整个上下文
         *      每次推理后立即返回
         **/
        let new_word_id: u32;

        let cache = history.cache();
        let mut binding = cache.write().unwrap();
        let kvcache = binding.deref_mut();

        let input_len = token_ids.len();

        if input_len >= max_len {
            // 此时，说明已经说满了，不能说更多了，所以……记录错误退吧
            return GenResult(u32::MAX, true, false, true);
        }

        // 每次的输入在外部控制，所以不用手动截断！
        let mut input = token_ids.to_vec();

        {
            let logits = self.forward(
                &Tensor::<u32>::new(input.clone(), &vec![input.len()]),
                kvcache,
            );

            // 我那不成器的TopPkT就注释掉留一个纪念吧……
            // let length = logits.size();
            // let data = unsafe { logits.data_mut() };
            // if temperature > 0. {
            //     for i in 0..length {
            //         data[i] /= temperature;
            //     }
            // }
            // let logits = easy_softmax(logits.data());
            // let new_word_id = Self::select_word_to_id(&logits.to_vec(), top_p, top_k as usize);

            new_word_id = random_sample(&logits, top_p, top_k, temperature);

            // // 调试-B
            // let tokenizer = GLOBAL_TOKEN.get().unwrap().read().unwrap();
            // flush_print!(
            //     "curr_word: [{}], kvc_len: {}\n",
            //     tokenizer.decode(&[new_word_id], true).unwrap(),
            //     kvcache.len()
            // );
            // // 调试-E
        }

        let kvc_len = kvcache.len();
        let pass_len = history.pass_len();
        let gen_len = kvc_len - pass_len;

        GenResult(
            new_word_id,
            gen_len == max_gen,
            new_word_id == self.eos_token_id,
            false,
        )
    }

    fn select_word_to_id(logits: &Vec<f32>, top_p: f32, top_k: usize) -> u32 {
        let mut indices_and_values: Vec<(_, _)> = logits
            .iter()
            .enumerate()
            .map(|(index, &value)| (index, value))
            .collect();

        indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut take_num = 1; // take_num 最小是1
        let mut tmp_sump = 0.;

        if top_p > 0. {
            for i in indices_and_values.iter() {
                tmp_sump += i.1;
                if tmp_sump > top_p {
                    break;
                }
                take_num += 1;
            }
        }

        take_num = if top_k > 0 {
            min(take_num, top_k)
        } else {
            take_num
        };

        let top_indices: Vec<f32> = indices_and_values
            .iter()
            .cloned()
            .take(take_num)
            .map(|(_, w)| w)
            .collect();

        let resampled = easy_softmax(&*top_indices);

        use rand::distributions::Distribution;
        use rand::distributions::WeightedIndex;

        let mut rng = rand::thread_rng();
        let index = WeightedIndex::new(resampled).unwrap().sample(&mut rng);

        indices_and_values[index].0 as u32
    }
}

fn self_attention(
    hidden_states: &mut Tensor<f32>, // (seq, n_kv_h * n_groups * dqkv)
    att_scores: &mut Tensor<f32>,    // (n_kv_h, n_groups, seq, total_seq)
    q: &Tensor<f32>,                 // (seq, n_kv_h * n_groups * dqkv)
    k: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    v: &Tensor<f32>,                 // (total_seq, n_kv_h * dqkv)
    n_kv_h: usize,                   // number of heads for k and v
    n_groups: usize,                 // n_head_of_q / n_head_of_k
    seq_len: usize,                  // input len of seq-token-vec
    total_seq_len: usize,            // len of kv-cache + seq_len
    dqkv: usize,                     // length of a single q, k, or v vector，人话：dim
) {
    // ### 以下是你需要实现的部分
    // score = Q @ K.T / sqrt(dim)
    // attn = softmax(score)
    // attn_V = attn @ V
    let bias: usize = seq_len * total_seq_len;
    let (beta, alpha) = (0f32, 1.0 / ((dqkv as f32).sqrt()));
    let (_m, _k, _n) = (seq_len, dqkv, total_seq_len);
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            // 这里的bh是针对scores的，对于QK不适用
            let bh = n_groups * head + group;
            // 此时取保证”假设“取出的Q和K是(seq_len, dim) 和 (total_seq_len, dim)
            let (_a, _b, _c) = (q.data(), k.data(), unsafe { att_scores.data_mut() });
            let clip_a = |i, j| {
                _a[i * n_kv_h * n_groups * _k + bh * _k + j] // 索引Q的节点
            };
            let clip_b = |i, j| {
                _b[i * n_kv_h * _k + head * _k + j] // 索引K的节点
            };
            // 完成索引后就是简单的求积之和
            for mi in 0.._m {
                for ni in 0.._n {
                    let idx = bh * bias + mi * total_seq_len + ni;
                    _c[idx] *= beta; // 乘0置零
                    _c[idx] += alpha
                        * (0.._k)
                            .map(|ki| clip_a(mi, ki) * clip_b(ni, ki))
                            .sum::<f32>();
                }
            }
        }
    }

    masked_softmax(att_scores);

    let (_a, _b, _c) = (att_scores.data(), v.data(), unsafe {
        hidden_states.data_mut()
    });
    for head in 0..n_kv_h {
        for group in 0..n_groups {
            // 这里开始算 attn_V = attn @ V =》 hidden = scores @ v
            let bh = n_groups * head + group;
            // 这里毕竟熟练了，就不写匿名函数索引了，直接写吧……
            for i in 0..seq_len {
                for j in 0..dqkv {
                    let idx = i * n_kv_h * n_groups * dqkv + bh * dqkv + j;
                    _c[idx] = (0..total_seq_len)
                        .map(|idx| {
                            _a[bh * bias + i * total_seq_len + idx]
                                * _b[head * dqkv + j + idx * n_kv_h * dqkv]
                        })
                        .sum::<f32>();
                }
            }
        }
    }
}

fn mlp(
    residual: &mut Tensor<f32>,      // x 以及之后的 y
    hidden_states: &mut Tensor<f32>, // 用于存储过程中的计算结果
    gate: &mut Tensor<f32>,
    up: &mut Tensor<f32>,
    w_up: &Tensor<f32>,
    w_down: &Tensor<f32>,
    w_gate: &Tensor<f32>,
    rms_w: &Tensor<f32>,
    eps: f32,
) {
    rms_norm(hidden_states, residual, rms_w, eps);
    matmul_transb(gate, 0f32, hidden_states, w_gate, 1f32);
    matmul_transb(up, 0f32, hidden_states, w_up, 1f32);
    swiglu(up, gate);
    matmul_transb(residual, 1f32, up, w_down, 1f32);
}
