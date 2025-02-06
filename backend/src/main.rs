/*
注意：
此项目使用了 夜间版(测试版) Rust，以及非正式的feature-coroutines构建生成器，所以标准版Rust会构建失败
另外，因为2024新标准对`static unsafe mut`的态度由`warn`->`error`所以，注意toml里的edition（不想改石山，懒得改了）
详见：https://doc.rust-lang.org/stable/unstable-book/language-features/coroutines.html
*/
#![feature(coroutines, coroutine_trait, stmt_expr_attributes)]

#[macro_use]
extern crate rocket;
mod config;
mod kvcache;
mod message;
mod model;
mod operators;
mod options;
mod params;
mod restful_api;
mod tensor;
mod chat_iter;

use crate::message::Message;
use crate::options::{inference, init_model, init_token};
use crate::restful_api::*;
use std::fs::read_to_string;
use std::path::PathBuf;

#[launch]
fn rocket() -> _ {
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("..").join("model");

    init_model(
        &*read_to_string(model_dir.clone().join("config.json")).unwrap(),
        std::fs::read(model_dir.clone().join("model.safetensors")).unwrap(),
    );

    init_token(&*read_to_string(model_dir.clone().join("tokenizer.json")).unwrap());

    // inference(
    //     "10086",
    //     "Hello!",
    //     |_| {},
    //     true,
    // );

    rocket::build().mount("/", routes![index, chat_ws])
}
