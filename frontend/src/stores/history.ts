// 此文件定义历史会话信息，就是一个uuid对应的all session集合
import {Session} from "./session.ts";

class History {
    sessions: Session[];
    uuid: string;
    name: string;
    is_loaded: boolean;

    constructor(uuid: string, name: string) {
        // 初始化的时候还未加载对话列表，所以is_loaded = False，在初次点进去才加载
        this.sessions = [];
        this.uuid = uuid;
        this.name = name;
        this.is_loaded = false;
    }

    *[Symbol.iterator]() {
        for (const i of this.sessions) {
            yield i;
        }
    }

    add_user_input(ipt: string) {
        this.sessions.push(new Session([ipt], "user"))
    }

    add_agent_input() {
        // 返回生成对象以及该追加的列表位置
        if (this.sessions.length > 0 && this.sessions[this.sessions.length-1].role == "assistant") {
            // 说明是重新生成的，所以使用追加
            this.sessions[this.sessions.length-1].msg.push("");
            this.sessions[this.sessions.length-1].gen();
            return [this.sessions[this.sessions.length-1], this.sessions[this.sessions.length-1].msg.length-1]
        }
        const tmp = new Session([""], "assistant");
        this.sessions.push(tmp);
        tmp.gen();
        return [tmp, 0];
    }

    push_agent_gen(idx: number, word: string) {
        // 向编号为idx的消息发送字符串，实际上就是add_agent_input返回的第二个参数
        let temp;
        if (this.sessions.length > 0 && this.sessions[this.sessions.length-1].role == "assistant") {
            temp = this.sessions[this.sessions.length-1].msg;
        } else {
            const tmp = new Session([""], "assistant");
            this.sessions.push(tmp);
            tmp.gen();
            temp = this.sessions[this.sessions.length-1].msg;
        }
        temp[idx] += word;
    }

    push_word_to_last(word: string) {
        // 直接向最后一个发送
        let temp;
        if (this.sessions.length > 0 && this.sessions[this.sessions.length-1].role == "assistant") {
            temp = this.sessions[this.sessions.length-1].msg;
        } else {
            const tmp = new Session([""], "assistant");
            this.sessions.push(tmp);
            tmp.gen();
            temp = this.sessions[this.sessions.length-1].msg;
        }
        temp[temp.length-1] += word;
    }

    close_last_gen() {
        if (this.sessions.length > 0 && this.sessions[this.sessions.length-1].role == "assistant") {
            this.sessions[this.sessions.length-1].close_gen();
        }
    }

    load() {
        // 还没有写网络请求，这里模拟一下
        this.is_loaded = true;
        // this.sessions.push(new Session(
        //     ["i am Leo"], "user"
        // ))
        // this.sessions.push(new Session(
        //     ["i am Assistant", "asdsdsadasdass", "QwQ QwQ"], "assistant"
        // ))
        // this.sessions.push(new Session(
        //     ["i am Leo"], "user"
        // ))
        // this.sessions.push(new Session(
        //     ["i am Assistant"], "assistant"
        // ))
        // this.sessions.push(new Session(
        //     ["i am Leo"], "user"
        // ))
        // this.sessions.push(new Session(
        //     ["i am Assistant"], "assistant"
        // ))
    }
}

export {History};