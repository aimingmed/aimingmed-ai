import { render, screen, fireEvent } from '@testing-library/react';
import App from './App';
import { vi } from 'vitest';

it('renders initial state', () => {
  render(<App />);
  expect(screen.getByText('Simple Chatbot')).toBeInTheDocument();
  expect(screen.getByRole('textbox')).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /send/i })).toBeInTheDocument();
});

// it('sends a message', () => {
//   const mockSend = vi.fn();
//   const mockSocket = { send: mockSend };
//   render(<App />);
//   const inputElement = screen.getByRole('textbox');
//   const buttonElement = screen.getByRole('button', { name: /send/i });
//   fireEvent.change(inputElement, { target: { value: 'Hello' } });
//   fireEvent.keyDown(inputElement, { key: 'Enter', code: 'Enter' }); // Simulate Enter key press
//   expect(mockSend).toHaveBeenCalled();
// });