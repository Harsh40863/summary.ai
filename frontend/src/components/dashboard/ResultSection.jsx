import React, { useCallback } from 'react';
import useSearch from '../../hooks/useSearch';
import useClipboard from '../../hooks/useClipboard';
import ResultCard from './ResultCard';

const ResultSection = () => {
  const { result, error } = useSearch();
  const { copy, isCopied } = useClipboard(2000);

  const handleCopy = useCallback((idx, text) => {
    copy(idx, text);
  }, [copy]);

  if (error) {
    return (
      <div className="result-container" style={{ marginTop: '20px' }}>
        <div className="upload-error" style={{ fontSize: '15px', padding: '14px 18px' }}>
          {error}
        </div>
      </div>
    );
  }

  if (!result || !result.results || result.results.length === 0) {
    if (result && (!result.results || result.results.length === 0)) {
      return (
        <div className="result-container result-card" style={{ marginTop: '20px' }}>
          <div className="result-label" style={{ color: '#A0A5B5' }}>
            No relevant matching sections found in your documents.
          </div>
        </div>
      );
    }
    return null;
  }

  return (
    <div className="result-container result-card">
      <div className="result-label">
        🔍 Found {result.results.length} relevant document{result.results.length > 1 ? 's' : ''}
      </div>

      {result.results.map((item, idx) => (
        <ResultCard
          key={`${item.document_name}-${idx}`}
          item={item}
          index={idx}
          onCopy={handleCopy}
          isCopied={isCopied(idx)}
        />
      ))}
    </div>
  );
};

export default React.memo(ResultSection);
