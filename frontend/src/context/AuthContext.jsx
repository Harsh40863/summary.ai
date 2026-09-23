import { createContext, useState, useEffect, useCallback } from 'react';
import api from '../utils/api';

export const AuthContext = createContext(null);

const TOKEN_KEY = 'hackindia_token';

// Helper to check token expiration safely
function isTokenValid(token) {
  if (!token) return false;
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    if (payload.exp && payload.exp * 1000 < Date.now()) {
      return false;
    }
    return true;
  } catch {
    return false;
  }
}

export const AuthProvider = ({ children }) => {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY));
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  const logout = useCallback(async () => {
    try {
      await api.post('/auth/logout');
    } catch (err) {
      console.warn('Logout API error:', err);
    } finally {
      localStorage.removeItem(TOKEN_KEY);
      setToken(null);
      setUser(null);
    }
  }, []);

  const refreshUser = useCallback(async () => {
    const currentToken = localStorage.getItem(TOKEN_KEY);
    if (!currentToken || !isTokenValid(currentToken)) {
      localStorage.removeItem(TOKEN_KEY);
      setToken(null);
      setUser(null);
      setAuthLoading(false);
      return;
    }

    try {
      const res = await api.get('/auth/me');
      setUser(res.data);
    } catch (err) {
      console.error('Failed to fetch current user:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem(TOKEN_KEY);
        setToken(null);
        setUser(null);
      }
    } finally {
      setAuthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  const login = useCallback(async (email, password) => {
    const res = await api.post('/auth/login', { email, password });
    const newToken = res.data.access_token;
    localStorage.setItem(TOKEN_KEY, newToken);
    setToken(newToken);
    await refreshUser();
    return res.data;
  }, [refreshUser]);

  const signup = useCallback(async (name, email, password) => {
    const res = await api.post('/auth/signup', { name, email, password });
    return res.data;
  }, []);

  const value = {
    user,
    token,
    isAuthenticated: !!token && isTokenValid(token),
    authLoading,
    login,
    signup,
    logout,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};
