import { createContext, useState, useCallback, useRef } from 'react';
import api from '../utils/api';

export const SearchContext = createContext(null);

export const SearchProvider = ({ children }) => {
  const [prompt, setPrompt] = useState('');
  const [action, setAction] = useState('search');
  const [language, setLanguage] = useState('en');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('Processing...');
  const [error, setError] = useState('');
  const [queryError, setQueryError] = useState('');
  const [shake, setShake] = useState(false);

  // In-memory Query Result Cache — prevents redundant AI API roundtrips
  const queryCache = useRef(new Map());
  const loadingTimerRef = useRef(null);

  const handleGenerate = useCallback(async () => {
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt) {
      setShake(true);
      setQueryError('Please enter a question first');
      setTimeout(() => {
        setShake(false);
        setQueryError('');
      }, 3000);
      return;
    }

    const cacheKey = `${trimmedPrompt.toLowerCase()}:::${action}:::${language}`;

    // Instant Cache Hit — 0ms delay!
    if (queryCache.current.has(cacheKey)) {
      setResult(queryCache.current.get(cacheKey));
      setError('');
      return;
    }

    setLoading(true);
    setError('');
    setResult(null);
    setLoadingMsg('Processing...');

    // Progressively update loading status for long-running AI inference
    loadingTimerRef.current = setTimeout(() => {
      setLoadingMsg('AI is thinking... (may take 30-60s on first run)');
    }, 5000);

    try {
      const res = await api.post('/api/query', {
        query: trimmedPrompt,
        action,
        threshold: 0.35,
        translate_to: language,
      });

      // Save into in-memory cache
      queryCache.current.set(cacheKey, res.data);
      setResult(res.data);
    } catch (err) {
      if (err.code === 'ECONNABORTED' || err.message?.includes('timeout')) {
        setError('⏱️ Request timed out. The AI is still warming up — please try again in 30 seconds.');
      } else if (err.response?.status === 502 || err.response?.status === 503 || err.response?.status === 504) {
        setError('🔄 Server is busy or waking up. Please wait 30 seconds and try again.');
      } else if (!err.response) {
        setError('🌐 Network error. Check your connection and try again.');
      } else {
        setError(err.response?.data?.detail || 'Query failed. Please try again.');
      }
    } finally {
      if (loadingTimerRef.current) {
        clearTimeout(loadingTimerRef.current);
      }
      setLoading(false);
      setLoadingMsg('Processing...');
    }
  }, [prompt, action, language]);

  const clearResult = useCallback(() => {
    setResult(null);
    setError('');
  }, []);

  const value = {
    prompt,
    setPrompt,
    action,
    setAction,
    language,
    setLanguage,
    result,
    setResult,
    loading,
    loadingMsg,
    error,
    setError,
    queryError,
    shake,
    handleGenerate,
    clearResult,
  };

  return (
    <SearchContext.Provider value={value}>
      {children}
    </SearchContext.Provider>
  );
};
