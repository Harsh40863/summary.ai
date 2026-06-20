import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
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
      // (those naturally return 401 on wrong credentials)
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

export default api;
