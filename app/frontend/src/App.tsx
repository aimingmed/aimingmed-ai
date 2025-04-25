import React, { useState, useEffect, useRef } from 'react';

const BASE_DOMAIN_NAME_PORT = import.meta.env.REACT_APP_DOMAIN_NAME_PORT || 'localhost:8004';


interface Message {
  sender: 'user' | 'bot';
  text: string;
}

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [socket, setSocket] = useState<WebSocket | null>(null);
  const [isLoading, setIsLoading] = useState(false);
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
          setMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            if (lastMessage && lastMessage.sender === 'bot') {
              return [...prevMessages.slice(0, -1), { ...lastMessage, text: lastMessage.text + data.payload }];
            } else {
              return [...prevMessages, { sender: 'bot', text: data.payload }];
            }
          });
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
    if (newMessage.trim() !== '') {
      setIsLoading(true);
      const message = [{ role: 'user', content: newMessage }];
      setMessages((prevMessages) => [...prevMessages, { sender: 'user', text: newMessage }]);
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
