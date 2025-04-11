import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
});

// https://vitest.dev/config/
export const test = {
  globals: true,
  environment: "jsdom",
  setupFiles: "./tests/setup.ts",
};
