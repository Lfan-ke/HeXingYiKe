// 此文件定义对话类，和Rust定义的Session一致
class Session {
    msg: string[];
    role: "user" | "assistant";
    is_gen: boolean;

    constructor(msg: string[], role: "user" | "assistant") {
        this.msg = msg;
        this.role = role;
        this.is_gen = false;
    }

    gen() {
        this.is_gen = true;
    }

    close_gen() {
        this.is_gen = false;
    }
}

export {Session};