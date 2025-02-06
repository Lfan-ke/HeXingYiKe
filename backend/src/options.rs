use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::message::{get_session, Message};
use crate::model::{GenResult, Llama};
use crate::operators::random_sample;
use crate::params::LLamaParams;
use crate::tensor::Tensor;
use once_cell::sync::{Lazy, OnceCell};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::ops::{Coroutine, DerefMut};
use crate::chat_iter::ChatIter;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use uuid::Uuid;
use crate::flush_print;

/// 模型最大处理多长，在模型初始化后就自动初始化了
pub static MAX_SEQ_LEN: OnceCell<Arc<RwLock<usize>>> = OnceCell::new();

pub static GLOBAL_LLAMA: OnceCell<Arc<RwLock<Llama<f32>>>> = OnceCell::new();
pub static GLOBAL_TOKEN: OnceCell<Arc<RwLock<Tokenizer>>> = OnceCell::new();

pub fn model_is_loaded() -> bool {
    // 模型是否已经加载完毕 - 非 none 约束
    !GLOBAL_LLAMA.get().is_none()
}

pub fn token_is_loaded() -> bool {
    // 是否已经加载完毕 - 非 none 约束
    !GLOBAL_TOKEN.get().is_none()
}

pub fn init_model(config_json: &str, model_file: Vec<u8>) {
    // 通过`json`的字符流加载`config`，以及字节流形式的`model`
    let config: LlamaConfigJson = serde_json::from_str(config_json).unwrap();
    let safetensor = SafeTensors::deserialize(&model_file).unwrap();
    let params = LLamaParams::from_safetensors(&safetensor, &config);

    let _ = MAX_SEQ_LEN.set(Arc::new(RwLock::new(config.max_position_embeddings)));

    unsafe {
        let _ = GLOBAL_LLAMA.set(Arc::new(RwLock::new(Llama {
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
        })));
    }
}

pub fn init_token(config_json: &str) {
    // 通过`json`的字符流加载`tokenizer`
    unsafe {
        GLOBAL_TOKEN
            .set(Arc::new(RwLock::new(
                serde_json::from_str(&config_json).unwrap(),
            )))
            .unwrap();
    }
}

pub(crate) fn serialization(input: &str) -> Vec<u32> {
    // 将 str 序列化为词表的 ids
    assert!(token_is_loaded());
    let tokenizer = GLOBAL_TOKEN.get().unwrap().read().unwrap().clone();
    let binding = tokenizer.encode(input, true).unwrap();
    let res = binding.get_ids();
    Vec::from(res)
}

pub(crate) fn deserialization(ids: &[u32]) -> String {
    // 将 ids 反序列化为 str
    assert!(token_is_loaded());
    let tokenizer = GLOBAL_TOKEN.get().unwrap().read().unwrap().clone();
    tokenizer.decode(&ids, true).unwrap()
}

fn new_cache() -> KVCache<f32> {
    assert!(model_is_loaded());
    let binding = GLOBAL_LLAMA.get().unwrap().clone();
    let model = binding.read().unwrap();
    model.new_cache()
}

/*  直接使用LocalStorage存KVC有点夸张，，，就单纯的存Texts吧……
 *  之后将历史数据同步之后，第一次推理时，再查看此会话是否在WASM已经有KVC，若有照常推理，若无则从头再来
 *  同理，前端管理会话的`delete/clean`与否，delete调用后端接口前后端同时进行删除，clean仅仅是后端删除
**/

pub(crate) struct History {
    // 历史多话题
    cache: Arc<RwLock<KVCache<f32>>>,
    // pass_len是上一次的total_seq_len，目前偷懒，只允许最后一次回滚（即只能回滚到上一个len）
    pass_len: Arc<RwLock<usize>>,
}

impl History {
    pub fn new() -> Self {
        Self {
            cache: Arc::new(RwLock::new(new_cache())),
            pass_len: Arc::new(RwLock::new(0)),
        }
    }

    /// 刷新缓存后，自己的历史记录也归零
    pub fn refresh_cache(&self) {
        self.cache.write().unwrap().refresh_cache();
        self.record_history();
    }

    pub fn pass_len(&self) -> usize {
        self.pass_len.read().unwrap().clone()
    }

    pub fn backtrack(&self) -> bool {
        if self.cache.clone().read().unwrap().len() == self.pass_len() {
            return false;
        }
        self.cache
            .clone()
            .write()
            .unwrap()
            .backtrack(self.pass_len());
        true
    }

    pub fn cache(&self) -> Arc<RwLock<KVCache<f32>>> {
        self.cache.clone()
    }

    pub fn record_history(&self) -> usize {
        // 记录当前cache的长度，返回记录的长度
        let mut ptr = self.pass_len.write().unwrap();
        // println!("record_history： 记录前长度：[{}]  记录后长度：[{}]", *ptr, self.cache.read().unwrap().len());
        *ptr = self.cache.read().unwrap().len();
        *ptr
    }

    pub fn cache_space(&self) -> usize {
        self.cache.read().unwrap().space()
    }
}

// uuid: history
pub static mut HISTORY: Lazy<HashMap<String, Arc<History>>> = Lazy::new(|| HashMap::new());

pub fn delete_history(uuid: &str) {
    // 删除HISTORY中的uuid对应
    unsafe {
        if HISTORY.get(&uuid.to_string()).is_some() {
            HISTORY.remove(&uuid.to_string());
        }
    }
}

pub fn contain_history(uuid: &str) -> bool {
    // 是否在历史记录里
    unsafe { HISTORY.contains_key(&uuid.to_string()) }
}

pub fn new_chat() -> String {
    // 使用 input 开启新一轮的对话，此时返回 uuid
    let uuid = Uuid::new_v4().to_string();
    unsafe {
        HISTORY.insert(uuid.clone(), Arc::new(History::new()));
    }
    uuid.clone()
}

pub fn new_chat_with_uuid(uuid: &str) -> String {
    // 使用 input 开启一段新的对话，且 uuid 是指定的，而不是随机的
    unsafe {
        HISTORY.insert(uuid.parse().unwrap(), Arc::new(History::new()));
    }
    uuid.to_string()
}

pub fn backtrack_chat(uuid: &str) -> bool {
    // 回溯一个话题，返回上一个（使用pass_len覆盖cache.len）
    // 但是如果已经回溯过，或者uuid对应的不存在，则会返回 false
    unsafe {
        if HISTORY.get(&uuid.to_string()).is_none() {
            return false;
        }
        let tmp = HISTORY.get(&uuid.to_string()).unwrap();
        tmp.backtrack()
    }
}

pub fn get_history(uuid: &str) -> Arc<History> {
    unsafe {
        match unsafe { HISTORY.get(&uuid.to_string()) } {
            None => {
                new_chat_with_uuid(uuid);
                HISTORY.get(&uuid.to_string()).unwrap().clone()
            }
            Some(history) => history.clone(),
        }
    }
}

/*  接下来是最重要的推理逻辑，传入 uuid 指定对话，如果此 uuid 不存在与 HISTORY 则会以此新建
 *  之后，若有前科，则继续逐词带入，如果没有前科，则先加上开始符号，推理一次之后再逐词带入
**/

/// 根据历史和上下文推理下一个词，如果生成满了，即：KVC大小超过最大序列长度就会返回None
/// :param max_len: 最大seq_len
/// :param max_gen: 最大允许生成长度
/// :param outer_message: 正在生成的MSG，一般Role是Ass，生成后会自动加入这里
/// 弃用
pub fn inference_next(
    context: &str,
    curr_history: Arc<History>,
    max_len: usize,
    max_gen: usize,
) -> GenResult {
    let model = GLOBAL_LLAMA.get().unwrap().read().unwrap();

    let mut ids = serialization(context);

    model.generate(&*ids, max_len, max_gen, 0.8, 30, 0.88, curr_history)
}

/// 历史会话的uuid，对话的整体上下文，每生成一个词的回调函数，是否本地命令行输出，只返回生成的输出（不附加结束符）
pub fn inference<T>(
    uuid: &str,
    message: &str,
    reader: T,
    shell_log: bool,
) -> Option<String>
where
    T: Fn(String),
{
    let mut res = vec![];
    let mut iter = ChatIter::new(uuid.into(), message.into(), 0.8, 10, 1.);
    while let Some((id, word)) = iter.next() {
        res.push(id);
        if shell_log {flush_print!("{} ", word)}
        reader(word.clone());
    }
    Option::from(deserialization(&res))
}

/// 前面两个，包括 generate 多多少少带点毛病，堆成石山代码了，所以用一个现代一点的方法吧
/// :param uuid: 会话的ID
/// :param user_msg: 用户的新输入
/// **弃用**：使用CRT.resume会报错QwQ
pub async fn async_inference(
    uuid: &str,
    user_msg: Message,
    top_p: f32,
    top_k: u32,
    temperature: f32,
) -> Box<dyn Coroutine<Return = (), Yield = ((u32, String))>+ '_> {
    // 如果没有溢出，那么生成器输入的tokens依然只有用户此次的输入
    // 倘若触发溢出，则会触发KVC截断与重载所有的消息文本
    // 会自动管理/申请/组织历史记录-KVC&MSG

    assert!(model_is_loaded());

    Box::new(#[coroutine]
    static move || {
        let lm_curr_history = get_history(uuid);
        let lm_curr_session = get_session(uuid);

        lm_curr_history.record_history();
        lm_curr_session.push_message(user_msg.clone());

        let max_len = MAX_SEQ_LEN.get().unwrap().read().unwrap().clone();
        let max_gen = max_len / 4;

        let generating_msg = Message::new_agent_msg("");

        let space = max_len - max_gen; // 至少空出 max_gen 的生成空间

        let mut last_space = max_gen;

        let cache_space = lm_curr_history.cache_space();

        let context = if cache_space > max_gen {
            // 用于输入的上下文信息
            // 需要截断KVC以及上下文即可
            lm_curr_history.refresh_cache();
            lm_curr_session.as_str_with_msg(&generating_msg, space)
        } else {
            // 不需要截断，所以要处理的只有用户输入和新消息
            lm_curr_history.record_history();
            lm_curr_session.as_str_only_2_msg_with_prompt(&user_msg, &generating_msg, space)
        };

        let cache = lm_curr_history.cache();
        let mut binding = cache.write().unwrap();
        let kvcache = binding.deref_mut();

        let token_ids = serialization(&*context);

        let mut input = token_ids.to_vec();

        let model = GLOBAL_LLAMA.get().unwrap().read().unwrap();

        let logits = model.forward(
            &Tensor::<u32>::new(input.clone(), &vec![input.len()]),
            kvcache,
        );
        let new_word_id = random_sample(&logits, top_p, top_k, temperature);

        if new_word_id == model.eos_token_id {
            generating_msg.set_is_over(true);
            lm_curr_session.push_message(generating_msg);
            return;
        }

        last_space -= 1;

        let word = deserialization(&[new_word_id]);

        generating_msg.push_word(&*word.clone());

        yield (new_word_id, word.clone());

        loop {
            input = vec![new_word_id];

            let logits = model.forward(
                &Tensor::<u32>::new(input.clone(), &vec![input.len()]),
                kvcache,
            );

            let new_word_id = random_sample(&logits, top_p, top_k, temperature);

            last_space -= 1;

            let word = deserialization(&[new_word_id]);

            generating_msg.push_word(&*word.clone());

            if new_word_id == model.eos_token_id || last_space == 0 {
                generating_msg.set_is_over(true);
                lm_curr_session.push_message(generating_msg);
                return;
            }

            yield (new_word_id, word.clone());
        }
    })
}
