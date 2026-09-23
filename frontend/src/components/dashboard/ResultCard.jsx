import React, { useMemo } from 'react';

function parseMarkdown(text) {
  if (!text) return { __html: '' };

  let html = text;
  // Only escape if it's not custom HTML from backend
  if (!text.includes('social-scroll-container') && !text.includes('social-post')) {
    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }

  html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #6C63FF;">$1</strong>');
  html = html.replace(/^[\s]*[\*\-]\s+(.*)$/gm, '<div style="margin-left: 16px; margin-bottom: 8px;">• $1</div>');

  // Format raw URLs into badges, avoiding URLs already inside href attribute
  html = html.replace(/(?<!["'])(https?:\/\/[^\s<]+)/g, '<a class="url-badge" href="$1" target="_blank" rel="noopener noreferrer">🔗 $1</a>');

  html = html.replace(/\n/g, '<br/>');
  html = html.replace(/<br\/>(<div style="margin-left: 16px)/g, '$1');
  html = html.replace(/<\/div><br\/>/g, '</div>');
  return { __html: html };
}

const ResultCard = ({ item, index, onCopy, isCopied }) => {
  const textToCopy = useMemo(() => {
    return (
      item.summary ||
      item.refined_insight ||
      (typeof item.web_content === 'string'
        ? item.web_content
        : JSON.stringify(item.web_content)) ||
      ''
    );
  }, [item]);

  // Memoize HTML parsing — zero regex recalculation on re-renders
  const formattedSummary = useMemo(() => parseMarkdown(item.summary), [item.summary]);
  const formattedInsight = useMemo(() => parseMarkdown(item.refined_insight), [item.refined_insight]);
  const formattedWebContent = useMemo(() => {
    if (!item.web_content) return null;
    return typeof item.web_content === 'string'
      ? parseMarkdown(item.web_content)
      : null;
  }, [item.web_content]);

  return (
    <div className="result-card-inner">
      <div className="result-card-header">
        <div className="result-doc-name">📄 {item.document_name}</div>
        <button
          className="btn-copy"
          onClick={() => onCopy(index, textToCopy)}
          type="button"
        >
          {isCopied ? (
            <span style={{ color: '#10B981' }}>Copied ✓</span>
          ) : (
            'Copy'
          )}
        </button>
      </div>
      <hr className="result-hr" />

      <div className="result-text-area">
        {item.summary && <div dangerouslySetInnerHTML={formattedSummary} />}
        {item.refined_insight && <div dangerouslySetInnerHTML={formattedInsight} />}
        {item.web_content && (
          <div>
            {typeof item.web_content === 'object' ? (
              <pre style={{ overflowX: 'auto', whiteSpace: 'pre-wrap' }}>
                {JSON.stringify(item.web_content, null, 2)}
              </pre>
            ) : (
              <div dangerouslySetInnerHTML={formattedWebContent} />
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default React.memo(ResultCard);
