import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { vi } from 'vitest';

it('renders initial state', () => {
  render(<App />);
  expect(screen.getByText('Simple Chatbot')).toBeInTheDocument();
  expect(screen.getByRole('textbox')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
});

it('sends a message', () => {
  const mockSend = vi.fn();
  vi.spyOn(WebSocket.prototype, 'send').mockImplementation(mockSend);
  render(<App />);
  const inputElement = screen.getByRole('textbox');
  fireEvent.change(inputElement, { target: { value: 'Hello' } });
  const buttonElement = screen.getByRole('button', { name: /send/i });
  fireEvent.click(buttonElement);
  expect(mockSend).toHaveBeenCalledWith(JSON.stringify([{ role: 'user', content: 'Hello' }]));
  expect(screen.getByText('Hello')).toBeInTheDocument();
});