mod utils; mod config; mod kvcache; mod model;
mod operators; mod params; mod tensor;

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use safetensors::SafeTensors;
use tokenizers::Tokenizer;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use crate::config::LlamaConfigJson;
use crate::kvcache::KVCache;
use crate::model::Llama;
use crate::params::LLamaParams;

static GLOBAL_LLAMA: Rc<RefCell<Option<Llama<f32>>>> = Rc::new(RefCell::new(None));
static GLOBAL_TOKEN: Rc<RefCell<Option<Tokenizer >>> = Rc::new(RefCell::new(None));

#[wasm_bindgen]
pub fn model_is_loaded() -> bool {
    // 模型是否已经加载完毕 - 非 none 约束
    ! GLOBAL_LLAMA.borrow().is_none()
}

#[wasm_bindgen]
pub fn token_is_loaded() -> bool {
    // 是否已经加载完毕 - 非 none 约束
    ! GLOBAL_TOKEN.borrow().is_none()
}

#[wasm_bindgen]
pub fn init_model(config_json: &str, model_file: Vec<u8>) {
    // 通过`json`的字符流加载`config`，以及字节流形式的`model`
    let config: LlamaConfigJson = serde_json::from_str(config_json).unwarp();
    let safetensor = SafeTensors::deserialize(&model_file).unwrap();
    let params = LLamaParams::from_safetensors(&safetensor, &config);

    unsafe {
        GLOBAL_LLAMA.borrow_mut().replace(Some(Llama {
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
        }));
    }
}

#[wasm_bindgen]
pub fn init_token(config_json: &str) {
    // 通过`json`的字符流加载`tokenizer`
    unsafe {
        GLOBAL_TOKEN.borrow_mut().replace(serde_json::from_str(&config_json).unwarp());
    }
}
//
// fn serialization(input: &str) -> &'static[u32] {
//     // 将 str 序列化为词表的 ids
//     assert!(token_is_loaded() && token_is_ready());
//     match GLOBAL_TOKEN.lock().unwrap().borrow().as_ref() {
//         None => { panic!() }
//         Some(tokenizer) => {
//             tokenizer.encode(input, true).unwrap().get_ids()
//         }
//     }
// }
//
// fn deserialization(ids: &[u32]) -> String {
//     // 将 ids 反序列化为 str
//     assert!(token_is_loaded() && token_is_ready());
//     match GLOBAL_TOKEN.lock().unwrap().borrow().as_ref() {
//         None => { panic!() }
//         Some(tokenizer) => {
//             tokenizer.decode(&ids, true).unwarp()
//         }
//     }
// }
//
// fn new_cache() -> KVCache<f32> {
//     assert!(model_is_loaded());
//     match GLOBAL_LLAMA.lock().unwrap().borrow().as_ref() {
//         None => {
//             panic!()
//         }
//         Some(model) => {
//             model.new_cache()
//         }
//     }
// }
//
// /*  直接使用LocalStorage存KVC有点夸张，，，就单纯的存Texts吧……
//  *  之后将历史数据同步之后，第一次推理时，再查看此会话是否在WASM已经有KVC，若有照常推理，若无则从头再来
//  *  同理，前端管理会话的`delete/clean`与否，delete调用后端接口前后端同时进行删除，clean仅仅是后端删除
// **/
//
// struct History {
//     // 历史多话题
//     cache: RefCell<KVCache<f32>>,
//     pass_len: usize,   // pass_len是上一次的total_seq_len，目前偷懒，只允许最后一次回滚（即只能回滚到上一个len）
// }
//
// impl History {
//     pub fn new() -> Self {
//         Self {
//             cache: RefCell::new(new_cache()),
//             pass_len: 0,
//         }
//     }
//
//     pub fn backtrack(&mut self) -> bool {
//         if self.cache.len() == self.pass_len {
//             return false
//         }
//         self.cache.get_mut().backtrack(self.pass_len);
//         true
//     }
// }
//
// static mut HISTORY: HashMap<&str, RefCell<History>> = HashMap::new();        // uuid: history
//
// #[wasm_bindgen]
// pub fn delete_history(uuid: &str) {
//     // 删除HISTORY中的uuid对应
//     unsafe {
//         if HISTORY.get(uuid).is_some() {
//             HISTORY.remove(uuid);
//         }
//     }
// }
//
// #[wasm_bindgen]
// pub fn contain_history(uuid: &str) -> bool {
//     // 是否在历史记录里
//     unsafe {
//         HISTORY.contains_key(uuid)
//     }
// }
//
// #[wasm_bindgen]
// pub fn new_chat() -> String {
//     // 使用 input 开启新一轮的对话，此时返回 uuid
//     let uuid = Uuid::new_v4().to_string();
//     unsafe {
//         HISTORY.insert(&*uuid, RefCell::new(History::new()));
//     }
//     uuid
// }
//
// #[wasm_bindgen]
// pub fn new_chat_with_uuid(uuid: &str) -> String {
//     // 使用 input 开启一段新的对话，且 uuid 是指定的，而不是随机的
//     unsafe {
//         HISTORY.insert(&*uuid, RefCell::new(History::new()));
//     }
//     uuid.to_string()
// }
//
// #[wasm_bindgen]
// pub fn backtrack_chat(uuid: &str) -> bool {
//     // 回溯一个话题，返回上一个（使用pass_len覆盖cache.len）
//     // 但是如果已经回溯过，或者uuid对应的不存在，则会返回 false
//     unsafe {
//         if HISTORY.get(uuid).is_none() { return false }
//         if let Some(mut history) = HISTORY.get(uuid) {
//             history.backtrack()
//         } else {
//             false
//         }
//     }
// }
//
// /*  接下来是最重要的推理逻辑，传入 uuid 指定对话，如果此 uuid 不存在与 HISTORY 则会以此新建
//  *  之后，若有前科，则继续逐词带入，如果没有前科，则先加上开始符号，推理一次之后再逐词带入
// **/
//
// #[wasm_bindgen]
// extern "C" {
//     fn yield_result(uuid: &str, word: &str, is_end: bool, is_full: bool);
// }
//
// #[wasm_bindgen]
// pub async fn inference(uuid: &str, context: &str) -> String {
//     // 使用context作为全部的上下文，使用 yield_result 进行异步传输生成的词
//     assert!(model_is_ready() && model_is_loaded());
//     let history = unsafe { HISTORY.get(uuid).unwrap().borrow() };
//     let cache = history.cache.get_mut();
//     match GLOBAL_LLAMA.lock().unwrap().borrow().as_ref() {
//         None => { panic!() }
//         Some(model) => {
//             deserialization(
//                 &*model.generate(
//                     serialization(context),
//                     500,
//                     0.8,
//                     30,
//                     1.,
//                     cache,
//                 )
//             )
//         }
//     }
// }
