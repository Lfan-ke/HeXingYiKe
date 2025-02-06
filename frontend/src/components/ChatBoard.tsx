/************************************************************
 * 这个组件是对话框，组成元素是一个板子 + 旁边使用伪元素的头像        *
 * 至于大模型对话的`Markdown`支持、图片、流程图等等支持，之后在说    *
 * 这个`Board`就是一个容器，之后有子组件后都可以按需生成放到此板子里  *
************************************************************/
import "./ChatBoard.less";
import {useState} from "react";

interface BoardProps {
    role: "user" | "assistant";
    msg: string[];
    is_gen?: boolean;       // 是否正在生成
    is_last?: boolean;      // 是否是最后一条
}

function Board(props: BoardProps) {
    const [currIdx, setCurrIdx] = useState(props.is_gen==true?props.msg.length-1:0);
    const syncSetCurrIdx = (index: number) => {
        // 在生成的时候锁死为最后一条消息
        setCurrIdx(props.is_gen==true?props.msg.length-1:index);
    }
    return (
        <div id="main-block-wrapper" className={props.role}>
            {/* 目前就俩角色，所以直接作为CSS名字方便一点 */}
            <div id="icon" className={props.role}>
                {/* 头像放左边旁边 */}
            </div>
            <div id="main-block" className={props.role}>
                <div id="main" className={props.role}>
                    {props.msg[currIdx]}
                </div>
            </div>
            <div id="page" className={props.role == "assistant" && props.msg.length > 1 ? undefined : "hidden"} title={props.is_gen?"生成中不允许换页":undefined}>
                {/* < 1/5 > 的翻历页定位放在右旁边 */}
                <span
                    onClick={() => syncSetCurrIdx(currIdx - 1 > 0 ? currIdx - 1 : 0)}>{currIdx == 0 ? "\u00A0\u00A0\u00A0\u00A0\u00A0" : "<\u00A0\u00A0\u00A0"}</span>
                {currIdx + 1}/{props.msg.length}
                <span
                    onClick={() => syncSetCurrIdx(currIdx + 1 < (props.msg.length - 1) ? currIdx + 1 : (props.msg.length - 1))}>{currIdx == props.msg.length - 1 ? "\u00A0\u00A0\u00A0\u00A0\u00A0" : "\u00A0\u00A0\u00A0>"}</span>
            </div>
            {
                props.is_last ?
                    (
                        <div id="hint" className={props.role == "user" ? "hidden" : undefined}>
                            {/* 停止生成/重新生成 的定位放左下角，最后一条才有 */}
                            <span>{props.is_gen ? <span id="generating">停止生成</span> : "重新生成"}</span>
                        </div>
                    ) : null
            }
        </div>
    )
}

export default Board;