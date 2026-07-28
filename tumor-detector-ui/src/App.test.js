import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the tumor detection heading', () => {
  render(<App />);
  expect(screen.getByText(/brain tumor image detection/i)).toBeInTheDocument();
});

test('renders the upload and auth controls', () => {
  render(<App />);
  expect(screen.getByRole('button', { name: /sign up/i })).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /log in/i })).toBeInTheDocument();
});
