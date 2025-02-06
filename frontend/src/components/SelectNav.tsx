import React from 'react';
import { History } from '../stores/history';

interface SelectNavProps {
    chats: Map<string, History>;
    setPage: (page: string) => void;
    curr_page: string;
}

const SelectNav: React.FC<SelectNavProps> = ({ chats, setPage, curr_page }) => {
    const chatItems = Array.from(chats.entries()).map(([uuid, history]) => (
        <div
            className={`history-item ${curr_page === uuid ? 'selected' : ''}`}
            key={uuid}
            onClick={() => setPage(uuid)}
        >
            {history.name}
        </div>
    ));

    return <>{chatItems}</>;
};

export default SelectNav;