import React from 'react';

function formatBytes(bytes) {
  if (!bytes) return '';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

const DocumentItem = ({ doc, isDeleting, onDelete }) => {
  return (
    <div className="doc-item">
      <div className="doc-filename">
        📄 {doc.name} {doc.content_length && <span className="doc-size">({formatBytes(doc.content_length)})</span>}
      </div>
      <button
        className={`btn-delete ${isDeleting ? 'deleting' : ''}`}
        onClick={() => onDelete(doc.id)}
        disabled={isDeleting}
        title="Delete document"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="3 6 5 6 21 6"></polyline>
          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
        </svg>
      </button>
    </div>
  );
};

export default React.memo(DocumentItem);
