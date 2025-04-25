import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const BASE_DOMAIN_NAME_PORT = import.meta.env.REACT_APP_DOMAIN_NAME_PORT || 'localhost:8004';

interface Message {
  sender: 'user' | 'bot';
  text: string;
}

interface ChatTurn {
  question: string;
  intermediateMessages: { title: string; payload: string }[];
  finalAnswer: string | null;
  isLoading: boolean;
  showIntermediate: boolean;
}

const App: React.FC = () => {
  const [chatTurns, setChatTurns] = useState<ChatTurn[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const mounted = useRef(false);

  // Disable input/button if any job is running
  const isJobRunning = chatTurns.some(turn => turn.isLoading);

  useEffect(() => {
    mounted.current = true;
    const ws = new WebSocket(`ws://${BASE_DOMAIN_NAME_PORT}/ws`);
    setSocket(ws);
    ws.onopen = () => {
      console.log('WebSocket connection opened');
    };
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setChatTurns((prevTurns) => {
          if (prevTurns.length === 0) return prevTurns;
          const lastTurn = prevTurns[prevTurns.length - 1];
          if (data.type === 'intermediate') {
            // Add intermediate message to the last turn
            const updatedTurn = {
              ...lastTurn,
              intermediateMessages: [...lastTurn.intermediateMessages, { title: data.title, payload: data.payload }],
            };
            return [...prevTurns.slice(0, -1), updatedTurn];
          } else if (data.type === 'final') {
            // Set final answer for the last turn
            const updatedTurn = {
              ...lastTurn,
              finalAnswer: data.payload,
            };
            return [...prevTurns.slice(0, -1), updatedTurn];
          } else if (data.type === 'done') {
            // Mark last turn as not loading
            const updatedTurn = {
              ...lastTurn,
              isLoading: false,
            };
            return [...prevTurns.slice(0, -1), updatedTurn];
          } else if (data.type === 'message' && data.payload && mounted.current) {
            // legacy support, treat as final
            const updatedTurn = {
              ...lastTurn,
              finalAnswer: (lastTurn.finalAnswer || '') + data.payload,
            };
            return [...prevTurns.slice(0, -1), updatedTurn];
          }
          return prevTurns;
        });
      } catch (error) {
        console.error('Error parsing message:', error);
      }
    };
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
    return () => {
      mounted.current = false;
      ws.close();
    };
  }, []);

  const sendMessage = () => {
    if (newMessage.trim() !== '') {
      setChatTurns((prev) => [
        ...prev,
        {
          question: newMessage,
          intermediateMessages: [],
          finalAnswer: null,
          isLoading: true,
          showIntermediate: false,
        },
      ]);
      const message = [{ role: 'user', content: newMessage }];
      socket?.send(JSON.stringify(message));
      setNewMessage('');
    }
  };

  const toggleShowIntermediate = (idx: number) => {
    setChatTurns((prev) => prev.map((turn, i) => i === idx ? { ...turn, showIntermediate: !turn.showIntermediate } : turn));
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="p-4">
        <h1 className="text-3xl font-bold text-center text-gray-800">Simple Chatbot</h1>
      </div>
      <div className="flex-grow overflow-y-auto p-4">
        {chatTurns.map((turn, idx) => (
          <React.Fragment key={idx}>
            {/* User question */}
            <div className="p-4 rounded-lg mb-2 bg-blue-100 text-blue-800">{turn.question}</div>
            {/* Status box for this question */}
            {turn.intermediateMessages.length > 0 && (
              <div className="mb-4">
                <div className="bg-blue-50 border border-blue-300 rounded-lg p-3 shadow-sm flex items-center">
                  {/* Spinner icon */}
                  {turn.isLoading && (
                    <svg className="animate-spin h-5 w-5 text-blue-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                    </svg>
                  )}
                  <span className="font-semibold text-blue-700 mr-2">Working on:</span>
                  {/* Key steps summary */}
                  <div className="flex flex-wrap gap-2">
                    {turn.intermediateMessages.map((msg, i) => (
                      <span key={i} className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-medium border border-blue-200">
                        {msg.title}
                      </span>
                    ))}
                  </div>
                  <button
                    className="ml-auto text-xs text-blue-600 flex items-center gap-1 px-2 py-1 rounded hover:bg-blue-100 focus:outline-none border border-transparent focus:border-blue-300 transition"
                    onClick={() => toggleShowIntermediate(idx)}
                    aria-expanded={turn.showIntermediate}
                    title={turn.showIntermediate ? 'Hide details' : 'Show details'}
                  >
                    <svg
                      className={`w-4 h-4 transition-transform duration-200 ${turn.showIntermediate ? 'rotate-180' : ''}`}
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>
                </div>
                {/* Expanded details */}
                {turn.showIntermediate && (
                  <div className="bg-white border border-blue-200 rounded-b-lg p-3 mt-1 text-xs max-h-64 overflow-y-auto">
                    {turn.intermediateMessages.map((msg, i) => (
                      <div key={i} className="mb-3">
                        <div className="font-bold text-blue-700 mb-1">{msg.title}</div>
                        <pre className="whitespace-pre-wrap break-words text-gray-800">{msg.payload}</pre>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
            {/* Final answer for this question */}
            {turn.finalAnswer && (
              <div className="prose p-4 rounded-lg mb-2 bg-gray-200 text-gray-800">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{turn.finalAnswer}</ReactMarkdown>              </div>
            )}
          </React.Fragment>
        ))}
      </div>
      <div className="p-4 border-t border-gray-300">
        <div className="flex">
          <input
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            className="flex-grow p-2 border border-gray-300 rounded-lg mr-2"
            disabled={isJobRunning}
          />
          <button 
            onClick={sendMessage} 
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg"
            disabled={isJobRunning}
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
