import React from 'react';
import useSearch from '../../hooks/useSearch';
import useDocuments from '../../hooks/useDocuments';

const QueryControls = () => {
  const {
    prompt,
    setPrompt,
    action,
    setAction,
    language,
    setLanguage,
    loading,
    loadingMsg,
    queryError,
    shake,
    handleGenerate,
  } = useSearch();

  const { languages } = useDocuments();

  const handleKeyDown = (e) => {
    // Cmd+Enter or Ctrl+Enter submits
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
      e.preventDefault();
      handleGenerate();
    }
  };

  return (
    <>
      <div className="dropdowns-row">
        <select
          className="dash-select"
          value={action}
          onChange={(e) => setAction(e.target.value)}
        >
          <option value="search">Search</option>
          <option value="explore">Explore (Web Enhanced)</option>
          <option value="think">Think (AI Insights)</option>
        </select>

        <select
          className="dash-select"
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
        >
          {Object.entries(languages).map(([code, langName]) => (
            <option key={code} value={code}>
              {langName}
            </option>
          ))}
          {Object.keys(languages).length === 0 && (
            <option value="en">English</option>
          )}
        </select>
      </div>

      <textarea
        className={`dash-textarea ${shake ? 'shake' : ''}`}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about your documents... (Press Cmd+Enter to search)"
      />
      {queryError && <div className="query-empty-error">{queryError}</div>}

      <button
        className="btn-generate"
        onClick={handleGenerate}
        disabled={loading}
      >
        {loading ? (
          <div className="loading-dots">
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="dot"></span>
            <span className="loading-label">{loadingMsg}</span>
          </div>
        ) : (
          'Generate Magic'
        )}
      </button>
    </>
  );
};

export default React.memo(QueryControls);
