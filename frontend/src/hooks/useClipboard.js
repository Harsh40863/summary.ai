import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * useClipboard hook provides robust copy-to-clipboard functionality
 * with automatic timeout resets and cleanup on unmount.
 */
export function useClipboard(timeout = 2000) {
  const [copiedKey, setCopiedKey] = useState(null);
  const timerRef = useRef(null);

  const copy = useCallback(async (key, text) => {
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);

      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }

      timerRef.current = setTimeout(() => {
        setCopiedKey(null);
      }, timeout);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  }, [timeout]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const isCopied = useCallback((key) => copiedKey === key, [copiedKey]);

  return { copy, isCopied, copiedKey };
}

export default useClipboard;
