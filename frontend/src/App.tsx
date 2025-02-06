import logo from "./assets/Hx1K.svg"
import "./App.less"

import {tmp_time} from "./utils/time.ts";
import {History as ChatHistory} from "./stores/history"

import {useState} from "react"
import { v4 as uuidv4 } from 'uuid';

import SelectNav from "./components/SelectNav.tsx";
import ChatArea from "./components/ChatArea.tsx";
import {CHAT} from "./fetch/config.ts";

function App() {
    // "new"永远代表new页面，当发送第一个user input的时候就会触发new接口
    const [chats, setChats] = useState(new Map<string, ChatHistory>());
    const [curr_page, setPage] = useState("new");

    function new_chat(): string {
        const uuid = uuidv4();
        chats.set(uuid, new ChatHistory(uuid, tmp_time("新对话")));
        setPage(uuid);
        setChats(new Map(chats));
        return uuid;
    }

    function sendInput() {
        const ipt: HTMLTextAreaElement | null = document.getElementById("input") as HTMLTextAreaElement;
        if (ipt && ipt.value.length > 0) {
            let uuid = ipt.dataset.uuid || "new";
            if (uuid == "new" || !chats.has(uuid)) {
                uuid = new_chat();
            }
            const user_input = ipt.value;
            ipt.value = ""; // 清空输入
            const history = chats.get(uuid);
            if (history===undefined) {return}
            history.add_user_input(user_input);
            history.add_agent_input();
            const socket = new WebSocket(`${CHAT}${uuid}`);
            socket.onopen = () => {
                socket.send(user_input);
            }
            socket.onmessage = (e) => {
                history.push_word_to_last(e.data=="\n"?"\n":`${e.data} `);
                setChats(new Map(chats));
            }
            socket.onclose = () => {
                history.close_last_gen();
                setChats(new Map(chats));
            }
            socket.onerror = (e) => {
                console.error(e);
                history.close_last_gen();
                setChats(new Map(chats));
                alert(`uuid: ${uuid} gen error!`);
            }
        }
    }

  return (
    < >
        <aside id="control-wrapper">
            <div id="control">
                <div id="header-wrapper">
                    <div id="header">
                        <div id="flex-logo">
                            <img id="logo" className="logo" src={logo} alt="禾心一可" data-meta="140 × 37 px"/>
                        </div>
                        <div id="flex-chat">
                            <div id="new-chat-btn" onClick={new_chat}>
                                新对话
                            </div>
                        </div>
                    </div>
                </div>
                <div id="navbar-wrapper">
                    <div id="navbar">
                        <div id="options-wrapper">
                            <div id="options">
                                <div id="search-wrapper">
                                    <input id="search" type="text" placeholder="搜索历史记录"/>
                                </div>
                                <button id="delete"></button>
                            </div>
                        </div>
                        <div id="history-wrapper">
                            <div id="history-container">
                                <SelectNav chats={chats} setPage={setPage} curr_page={curr_page} />
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </aside>
        {/* 用注释分割一下，不然会看花眼QwQ */}
        <main  id="chatpkg-wrapper">
            <div id="chatpkg">
                <div id="decoration">
                    {/* 顶部fix一个光影装饰 */}
                </div>
                <div id="chat-container">
                    {/* 这里是对话的容器，中间是对话和LM生成文本的组件*/}
                    <ChatArea chats={chats} curr_page={curr_page} setChats={setChats} />
                </div>
                <div id="user-ipt-wrapper">
                    <div id="text-area-wrapper">
                        <textarea id="input" placeholder="Talk something ..." data-uuid={curr_page}></textarea>
                        <button id="sender" className={"v2"} onClick={sendInput}>
                            {/* v1是大一点的，好看但是和设计稿有点出入 */}
                            {/* v2是按着设计稿来的，就是因为出来后不太好看 */}
                        </button>
                    </div>
                    <div id="copyright">
                        本页面搭载`Learn-LM-Rs`提供的`Chat`模型，内容由AI生成，无法保证真实准确，仅供参考，网页属于临摹作品，只供完成进阶阶段作业
                    </div>
                </div>
            </div>
        </main >
    </>
  )
}

export default App
