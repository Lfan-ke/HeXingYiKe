// 这里利用传入的消息来动态渲染ChatBoards
import ChatBoard from "./ChatBoard.tsx"
import { History } from "../stores/history.ts";
import React from "react";
import {tmp_time} from "../utils/time.ts";

interface ChatAreaProps {
    chats: Map<string, History>;
    curr_page: string;
    setChats: (chats: Map<string, History>) => void;
}

const ChatArea: React.FC<ChatAreaProps> = ({ chats, curr_page, setChats }) => {
    let session = chats.get(curr_page);
    if (session === undefined) {
        session = new History(curr_page, tmp_time("新对话"));
        chats.set(curr_page, session);
        setChats(new Map<string, History>(chats));
    }
    if (!session.is_loaded) { session.load() }
    const boards = [];
    for (const item of session) {
        boards.push(<ChatBoard key={boards.length} role={item.role} msg={item.msg} is_gen={item.is_gen} is_last={boards.length+1 == session.sessions.length} />);
    }

    return <>{boards}</>;
};

export default ChatArea;
