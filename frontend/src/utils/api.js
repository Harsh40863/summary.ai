import axios from 'axios';

// Call Railway DIRECTLY — bypasses Vercel's 30s proxy timeout
// Railway URL is not secret since we use JWT Bearer tokens for security
const RAILWAY_URL = 'https://summaryai-production-bc85.up.railway.app';

const api = axios.create({
  baseURL: RAILWAY_URL,
  timeout: 120000, // 2 minute timeout — ML operations (search/think) can take time
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
  axios.get(`${RAILWAY_URL}/ping`).catch(() => {});
  // Then every 4 minutes
  keepAliveInterval = setInterval(() => {
    axios.get(`${RAILWAY_URL}/ping`).catch(() => {});
  }, 4 * 60 * 1000);
}

export function stopKeepAlive() {
  if (keepAliveInterval) {
    clearInterval(keepAliveInterval);
    keepAliveInterval = null;
  }
}

export default api;
