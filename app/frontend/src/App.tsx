import React, { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';

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
  const [messages, setMessages] = useState<Message[]>([]);
  const [intermediateMessages, setIntermediateMessages] = useState<{title: string, payload: string}[]>([]);
  const [finalAnswer, setFinalAnswer] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showIntermediate, setShowIntermediate] = useState(false);
  const [newMessage, setNewMessage] = useState('');
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const mounted = useRef(false);

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
                if (data.type === 'message' && data.payload && mounted.current) {
                    // legacy support, treat as final
                    setMessages((prevMessages) => {
                        const lastMessage = prevMessages[prevMessages.length - 1];
                        if (lastMessage && lastMessage.sender === 'bot') {
                            return [...prevMessages.slice(0, -1), { ...lastMessage, text: lastMessage.text + data.payload }];
                        } else {
                            return [...prevMessages, { sender: 'bot', text: data.payload }];
                        }
                    });
                    setFinalAnswer(data.payload);
                } else if (data.type === 'intermediate') {
                    setIntermediateMessages((prev) => [...prev, { title: data.title, payload: data.payload }]);
                } else if (data.type === 'final') {
                    setFinalAnswer(data.payload);
                } else if (data.type === 'done') {
                    setIsLoading(false);
                } else {
                    console.error('Unexpected message format:', data);
                }
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
    if (newMessage.trim() !== '' && !isLoading) {
      setIsLoading(true);
      setIntermediateMessages([]);
      setFinalAnswer(null);
      setMessages((prevMessages) => [...prevMessages, { sender: 'user', text: newMessage }]);
      const message = [{ role: 'user', content: newMessage }];
      socket?.send(JSON.stringify(message));
      setNewMessage('');
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      <div className="p-4">
        <h1 className="text-3xl font-bold text-center text-gray-800">Simple Chatbot</h1>
      </div>
      <div className="flex-grow overflow-y-auto p-4">
        {messages.map((msg, index) => (
          <div key={index} className={`p-4 rounded-lg mb-2 ${msg.sender === 'user' ? 'bg-blue-100 text-blue-800' : 'bg-gray-200 text-gray-800'}`}>
            {msg.text}
          </div>
        ))}
        {/* Status box for intermediate steps */}
        {intermediateMessages.length > 0 && (
          <div className="mb-4">
            <div className="bg-blue-50 border border-blue-300 rounded-lg p-3 shadow-sm flex items-center">
              {/* Spinner icon */}
              {isLoading && (
                <svg className="animate-spin h-5 w-5 text-blue-500 mr-2" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"></path>
                </svg>
              )}
              <span className="font-semibold text-blue-700 mr-2">Working on:</span>
              {/* Key steps summary */}
              <div className="flex flex-wrap gap-2">
                {intermediateMessages.map((msg, idx) => (
                  <span key={idx} className="bg-blue-100 text-blue-700 px-2 py-1 rounded text-xs font-medium border border-blue-200">
                    {msg.title}
                  </span>
                ))}
              </div>
              <button
                className="ml-auto text-xs text-blue-600 underline focus:outline-none"
                onClick={() => setShowIntermediate((v) => !v)}
              >
                {showIntermediate ? 'Hide details' : 'Show details'}
              </button>
            </div>
            {/* Expanded details */}
            {showIntermediate && (
              <div className="bg-white border border-blue-200 rounded-b-lg p-3 mt-1 text-xs max-h-64 overflow-y-auto">
                {intermediateMessages.map((msg, idx) => (
                  <div key={idx} className="mb-3">
                    <div className="font-bold text-blue-700 mb-1">{msg.title}</div>
                    <pre className="whitespace-pre-wrap break-words text-gray-800">{msg.payload}</pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
        {/* Final answer for this question */}
        {finalAnswer && (
          <div className="p-4 rounded-lg mb-2 bg-gray-200 text-gray-800 prose max-w-none">
            <ReactMarkdown>{finalAnswer}</ReactMarkdown>
          </div>
        )}
      </div>
      <div className="p-4 border-t border-gray-300">
        <div className="flex">
          <input
            type="text"
            value={newMessage}
            onChange={(e) => setNewMessage(e.target.value)}
            className="flex-grow p-2 border border-gray-300 rounded-lg mr-2"
            disabled={isLoading}
          />
          <button onClick={sendMessage} className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg" disabled={isLoading}>
            {isLoading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default App;
