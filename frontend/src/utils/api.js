import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || (typeof window !== 'undefined' && (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') ? 'http://localhost:8000' : 'https://summaryai-production-bc85.up.railway.app');

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000, // 2 minute timeout — ML operations (search/think) can take time
  withCredentials: true,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('hackindia_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
}, (error) => Promise.reject(error));

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      // Only auto-logout if the failed request was NOT a login/signup call
      const requestUrl = error.config?.url || '';
      const isAuthCall = requestUrl.includes('/auth/login') || requestUrl.includes('/auth/signup');
      const isOnProtectedPage = !window.location.pathname.includes('/login') && !window.location.pathname.includes('/signup') && window.location.pathname !== '/';
      
      if (!isAuthCall && isOnProtectedPage) {
        localStorage.removeItem('hackindia_token');
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

// ─── Keep-alive ping ──────────────────────────────────────────────────────────
// Pings Railway every 4 minutes to prevent cold starts (Railway sleeps after 5min idle)
let keepAliveInterval = null;

export function startKeepAlive() {
  if (keepAliveInterval) return; // already running
  // Ping immediately
  axios.get(`${API_BASE_URL}/ping`).catch(() => {});
  // Then every 4 minutes
  keepAliveInterval = setInterval(() => {
    axios.get(`${API_BASE_URL}/ping`).catch(() => {});
  }, 4 * 60 * 1000);
}

export function stopKeepAlive() {
  if (keepAliveInterval) {
    clearInterval(keepAliveInterval);
    keepAliveInterval = null;
  }
}

export default api;
