use crate::message::{get_session, Message as LmMsg};
use crate::model::GenResult;
use crate::options::{
    deserialization, get_history, inference, inference_next, model_is_loaded, MAX_SEQ_LEN,
};
use crate::rocket::futures::SinkExt;
use crate::rocket::futures::StreamExt;
use rocket::async_stream::stream;
use serde_json::de::Read;
use std::char::decode_utf16;
use std::iter::Iterator;
use std::thread;
use ws::{Stream, WebSocket};
use crate::chat_iter::ChatIter;
use crate::flush_print;

///除了一个`hello world`的接口外，其余的接口：
/// -   历史对话记录
///     -   增 -> 返回 uuid
///     -   删 -> 返回 uuid
///     -   回溯 -> 返回 uuid
///     -   查 -> 传入一个 uuid 查询出整个对话的 Json
/// -   对话
///     -   传入 uuid 打开 socket 进行会话
///     -   即使会话断开也会继续生成，直到结束
///     -   滑动窗口轮换：当生成剩余空间不足，会通知前端，且在本地进行滑动窗口，但是仍然会保存整个对话历史
/// emm，就先这些算了……

#[get("/")]
pub fn index() -> &'static str {
    "Hello, world!"
}

/// 开启一个新的会话，返回 uuid 并开启一个会话
#[get("/c/<uuid>")]
pub fn chat_ws(uuid: String, ws: WebSocket) -> ws::Channel<'static> {
    // 和Local推理的 inference 一样……
    ws.channel(move |mut stream| {
        Box::pin(async move {
            while let Some(message) = stream.next().await {
                let message = message.unwrap();
                if message.is_text() {
                    // 文本输入的信息
                    let user_input = message.into_text().unwrap();
                    let mut iter = ChatIter::new(uuid.clone().into(), user_input.into(), 0.8, 10, 1.);
                    while let Some((id, word)) = iter.next() {
                        stream.send(word.into()).await.unwrap();
                    }
                    // 暂时用完就关闭
                    stream.close(None).await.unwrap();
                } else if message.is_close() {
                    // 关闭……
                } else if message.is_binary() {
                    // 二进制的信息特地的停止生成并回溯
                }
            }
            return Ok(());
        })
    })
}
