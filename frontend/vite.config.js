import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /auth and /api requests to http://localhost:8000
// This hides the backend URL from the browser completely
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/auth': 'http://localhost:8000',
      '/api': 'http://localhost:8000'
    }
  }
})
