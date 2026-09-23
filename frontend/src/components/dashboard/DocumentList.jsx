import React, { useCallback } from 'react';
import useDocuments from '../../hooks/useDocuments';
import DocumentItem from './DocumentItem';

const DocumentList = () => {
  const { documents, deletingId, deleteDocument } = useDocuments();

  const handleDelete = useCallback((id) => {
    deleteDocument(id);
  }, [deleteDocument]);

  return (
    <div className="doc-list">
      {documents.length === 0 ? (
        <div className="empty-docs">No documents yet. Upload one above.</div>
      ) : (
        documents.map((doc, idx) => (
          <DocumentItem
            key={doc.id || doc._id || idx}
            doc={doc}
            isDeleting={deletingId === (doc.id || doc._id)}
            onDelete={handleDelete}
          />
        ))
      )}
    </div>
  );
};

export default React.memo(DocumentList);
