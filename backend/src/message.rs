use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, RwLock};

pub enum Role {
    User,
    Assistant,
}

impl Role {
    const USER_STR: &'static str = "user";
    const ASSISTANT_STR: &'static str = "assistant";

    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => Role::USER_STR,
            Role::Assistant => Role::ASSISTANT_STR,
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            Role::USER_STR => Some(Role::User),
            Role::ASSISTANT_STR => Some(Role::Assistant),
            _ => None,
        }
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Role::User => write!(f, "{}", Role::USER_STR),
            Role::Assistant => write!(f, "{}", Role::ASSISTANT_STR),
        }
    }
}

/// 一个保存会话结构的结构体
pub struct Session {
    history: Arc<RwLock<Vec<Message>>>,
    prompt: Arc<RwLock<Prompt>>,
    name: Arc<RwLock<String>>,
}

impl Session {
    pub fn new() -> Self {
        Self {
            history: Arc::new(Default::default()),
            prompt: Arc::new(RwLock::new(Prompt::new(""))),
            name: Arc::new(Default::default()),
        }
    }

    pub fn new_with_prompt(prompt: &str) -> Self {
        Self {
            history: Arc::new(Default::default()),
            prompt: Arc::new(RwLock::new(Prompt::new(prompt.clone()))),
            name: Arc::new(Default::default()),
        }
    }

    /// 生成让AI看的模板字符串，且会考虑最大长度
    pub fn as_str(&self, max_length: usize) -> String {
        let mut tmp = String::new();
        let prompt = self.prompt.read().unwrap();
        let history = self.history.read().unwrap();
        for msg in history.iter() {
            tmp += &msg.as_str()
        }
        tmp += &*prompt.as_str();
        if tmp.len() > max_length {
            let tct_begin = tmp.len() - max_length;
            let truncated: &str = &tmp[tct_begin..];
            tmp = truncated.into();
        }
        tmp.as_str().into()
    }

    /// 暂时联合外界的一个MSG共同生成字符串
    pub fn as_str_with_msg(&self, msg: &Message, max_length: usize) -> String {
        let mut tmp = String::new();
        let prompt = self.prompt.read().unwrap();
        let history = self.history.read().unwrap();
        for msg in history.iter() {
            tmp += &msg.as_str()
        }
        tmp += &*prompt.as_str();
        tmp += &*msg.as_str();
        if max_length <= tmp.len() {
            let tct_begin = tmp.len() - max_length;
            let truncated: &str = &tmp[tct_begin..];
            tmp = truncated.into();
        }
        tmp.as_str().into()
    }

    pub fn as_str_only_2_msg_with_prompt(
        &self,
        msg1: &Message,
        msg2: &Message,
        max_length: usize,
    ) -> String {
        let mut tmp = String::new();
        tmp += &msg1.as_str();
        tmp += &*self.prompt.read().unwrap().as_str();
        tmp += &msg2.as_str();
        if max_length <= tmp.len() {
            let truncated: &str = &tmp[tmp.len() - max_length..];
            tmp = truncated.into();
        }
        tmp.as_str().into()
    }

    // /// 从模板字符串生成对话信息，比如|<xxx>|xxx之后写吧
    // fn from_str(s: &str) -> Option<Self> {
    // }

    pub fn push_message(&self, msg: Message) {
        let mut history = self.history.write().unwrap();
        history.push(msg);
    }

    /// 回溯一个话题，因为需要回溯的永远是AI说的，所以如果Role是AI则删除，否则就不管（最后一条是用户说明回溯过了）
    pub fn backtrack(&self) -> bool {
        let mut history = self.history.write().unwrap();
        match history.pop() {
            None => false,
            Some(msg) => match msg.role {
                Role::User => {
                    history.push(msg);
                    false
                }
                Role::Assistant => true,
            },
        }
    }
}

impl fmt::Display for Session {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut tmp = String::new();
        let prompt = self.prompt.read().unwrap();
        let history = self.history.read().unwrap();
        for msg in history.iter() {
            tmp += &msg.as_str()
        }
        tmp += &*prompt.as_str();
        write!(f, "{}", tmp.as_str())
    }
}

/// 一条消息
pub struct Message {
    role: Role,
    content: Arc<RwLock<String>>,
    is_over: Arc<RwLock<bool>>,
}

impl Message {
    pub fn clone(&self) -> Self {
        let role = self.role.as_str();
        Self {
            role: Role::from_str(role).unwrap(),
            content: Arc::new(RwLock::new(self.content.read().unwrap().clone())),
            is_over: Arc::new(RwLock::new(self.is_over())),
        }
    }
}

impl Message {
    pub fn new_agent_msg(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: Arc::new(RwLock::new(content.into())),
            is_over: Arc::new(RwLock::new(false)), // agent的默认是false
        }
    }

    pub fn new_user_msg(content: &str) -> Self {
        Self {
            role: Role::User,
            content: Arc::new(RwLock::new(content.into())),
            is_over: Arc::new(RwLock::new(true)), // user 的默认是 true
        }
    }

    pub fn as_str(&self) -> String {
        let mut head = format!(
            "<|im_start|>{} {}\n",
            self.role,
            self.content.read().unwrap()
        )
        .as_str()
        .into();
        if self.is_over() {
            head += "<|im_end|>\n";
        }
        head
    }

    pub fn push_word(&self, word: &str) {
        let mut content = self.content.write().unwrap();
        content.push_str(format!(" {}", word).as_str()); // 因为 Chat 模型的尿性，得前置一个空格
    }

    pub fn set_is_over(&self, is_over: bool) {
        let mut content = self.is_over.write().unwrap();
        *content = is_over;
    }

    pub fn is_over(&self) -> bool {
        self.is_over.read().unwrap().clone()
    }
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

/// 存“洗脑”信息
pub struct Prompt {
    content: String,
}

impl Prompt {
    pub fn new(content: &str) -> Self {
        Self {
            content: content.to_string(),
        }
    }

    pub fn as_str(&self) -> String {
        format!("{}", self.content.as_str()).as_str().clone().into()
    }
}

impl fmt::Display for Prompt {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.as_str().fmt(f)
    }
}

/// 本地管理会话用，uuid: Session
static mut SESSION: Lazy<HashMap<String, Arc<Session>>> = Lazy::new(|| HashMap::new());

// 下面是使用 uuid 管理Map的示例

pub fn delete_session(uuid: &str) {
    // 删除HISTORY中的uuid对应
    unsafe {
        if SESSION.get(&uuid.to_string()).is_some() {
            SESSION.remove(&uuid.to_string());
        }
    }
}

pub fn contain_session(uuid: &str) -> bool {
    // 是否在历史记录里
    unsafe { SESSION.contains_key(&uuid.to_string()) }
}

pub fn new_session_with_uuid(uuid: &str) -> String {
    // 使用 input 开启一段新的对话，且 uuid 是指定的，而不是随机的
    unsafe {
        SESSION.insert(uuid.parse().unwrap(), Arc::new(Session::new()));
    }
    uuid.to_string()
}

pub fn backtrack_session(uuid: &str) -> bool {
    // 回溯一个话题，因为需要回溯的永远是AI说的，所以如果Role是AI则删除，否则就不管（最后一条是用户说明回溯过了）
    if unsafe { SESSION.get(&uuid.to_string()).is_none() } {
        return false;
    }
    let tmp = unsafe { SESSION.get(&uuid.to_string()).unwrap() };
    tmp.backtrack()
}

pub fn get_session(uuid: &str) -> Arc<Session> {
    unsafe {
        match SESSION.get(&uuid.to_string()) {
            None => {
                new_session_with_uuid(uuid);
                SESSION.get(&uuid.to_string()).unwrap().clone()
            }
            Some(session) => session.clone(),
        }
    }
}
