注意启用了`#![feature(generators, generator_trait)]`

这导致得启用预发布编译
```shell
rustup install nightly
rustup default nightly
# 如果没有把预发布版本设置为默认：cargo +nightly build
```
