/***
因为Feature和Coroutine的失败，所以独立一个类使用迭代器代替生成器的思路来吧……
    // let mut generator = async_inference(uuid, message, 0.8, 30, 0.88);
    // let b = generator;
    // let c = pin!(b);
    // println!("{:?}", c.resume(()));
上面代码以示纪念………………
***/
use std::ops::DerefMut;
use crate::message::{get_session, Message};
use crate::operators::random_sample;
use crate::options::{deserialization, get_history, model_is_loaded, GLOBAL_LLAMA, MAX_SEQ_LEN};
use crate::tensor::Tensor;

pub struct ChatIter {
    uuid: String,
    user_msg: Message,
    top_p: f32,
    top_k: u32,
    temperature: f32,
    curr_id: Option<u32>,
    gen_len: usize,
}

impl ChatIter {
    pub fn new(
        uuid: String,
        user_msg: String,
        top_p: f32,
        top_k: u32,
        temperature: f32,
    ) -> Self {
        Self {
            uuid, top_p, top_k, temperature,
            user_msg: Message::new_user_msg(&user_msg), curr_id: None, gen_len: 0,
        }
    }
}

impl Iterator for ChatIter {
    type Item = (u32, String);

    fn next(&mut self) -> Option<Self::Item> {

        assert!(model_is_loaded());

        let lm_curr_history = get_history(&*self.uuid);
        let lm_curr_session = get_session(&*self.uuid);

        lm_curr_history.record_history();
        lm_curr_session.push_message(self.user_msg.clone());

        let max_len = MAX_SEQ_LEN.get().unwrap().read().unwrap().clone();
        let max_gen = max_len / 4;

        let generating_msg = Message::new_agent_msg("");

        let space = max_len - max_gen; // 至少空出 max_gen 的生成空间

        let mut last_space = max_gen;

        let cache_space = lm_curr_history.cache_space();

        let mut input;

        if self.gen_len == 0 {
            let context = if cache_space > max_gen {
                // 用于输入的上下文信息，需要截断KVC以及上下文即可
                lm_curr_history.refresh_cache();
                lm_curr_session.as_str_with_msg(&generating_msg, space)
            } else {
                // 不需要截断，所以要处理的只有用户输入和新消息
                lm_curr_history.record_history();
                lm_curr_session.as_str_only_2_msg_with_prompt(&self.user_msg, &generating_msg, space)
            };
            let token_ids = crate::options::serialization(&*context);

            input = token_ids.to_vec();
        } else {
            input = [self.curr_id.unwrap()].to_vec();
        }

        let cache = lm_curr_history.cache();
        let mut binding = cache.write().unwrap();
        let kvcache = binding.deref_mut();

        let model = GLOBAL_LLAMA.get().unwrap().read().unwrap();

        let logits = model.forward(
            &Tensor::<u32>::new(input.clone(), &vec![input.len()]),
            kvcache,
        );
        let new_word_id = random_sample(&logits, self.top_p, self.top_k, self.temperature);

        if new_word_id == model.eos_token_id {
            generating_msg.set_is_over(true);
            lm_curr_session.push_message(generating_msg);
            return None;
        }

        last_space -= 1; self.gen_len += 1;

        let word = deserialization(&[new_word_id]);

        generating_msg.push_word(&*word.clone());

        self.curr_id = Some(new_word_id);

        Some((new_word_id, word.clone()))
    }
}