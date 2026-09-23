import { createContext, useState, useEffect, useCallback, useRef } from 'react';
import api from '../utils/api';

export const DocumentContext = createContext(null);

export const DocumentProvider = ({ children }) => {
  const [documents, setDocuments] = useState([]);
  const [loadingDocs, setLoadingDocs] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [error, setError] = useState('');
  const [languages, setLanguages] = useState({});

  const languagesFetchedRef = useRef(false);

  const fetchDocuments = useCallback(async () => {
    setLoadingDocs(true);
    try {
      const res = await api.get('/api/documents');
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    } finally {
      setLoadingDocs(false);
    }
  }, []);

  const fetchLanguages = useCallback(async () => {
    // Cache check to avoid redundant network calls
    if (languagesFetchedRef.current && Object.keys(languages).length > 0) return;
    try {
      const res = await api.get('/api/languages');
      setLanguages(res.data.languages || {});
      languagesFetchedRef.current = true;
    } catch (err) {
      console.error('Failed to fetch languages:', err);
    }
  }, [languages]);

  const uploadFiles = useCallback(async (files) => {
    if (!files || files.length === 0) return { success: false };
    setUploading(true);
    setUploadMsg('');
    setError('');

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const res = await api.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const msg = res.data.message || 'Files uploaded successfully!';
      setUploadMsg(msg);
      await fetchDocuments();
      return { success: true, message: msg };
    } catch (err) {
      const errMsg = err.response?.data?.detail || 'Upload failed. Please try again.';
      setError(errMsg);
      return { success: false, error: errMsg };
    } finally {
      setUploading(false);
    }
  }, [fetchDocuments]);

  // Optimistic Document Deletion — Zero UI Delay
  const deleteDocument = useCallback(async (docId) => {
    setDeletingId(docId);
    setError('');

    // Snapshot current state for rollback if deletion fails
    let previousDocs;
    setDocuments((prev) => {
      previousDocs = prev;
      return prev.filter((doc) => doc.id !== docId);
    });

    try {
      await api.delete(`/api/documents/${docId}`);
      setDeletingId(null);
      return { success: true };
    } catch (err) {
      console.error('Failed to delete document, rolling back optimistic update:', err);
      // Rollback to previous state
      if (previousDocs) {
        setDocuments(previousDocs);
      }
      setError(err.response?.data?.detail || 'Failed to delete document. Please try again.');
      setDeletingId(null);
      return { success: false, error: err };
    }
  }, []);

  const clearMessages = useCallback(() => {
    setUploadMsg('');
    setError('');
  }, []);

  const value = {
    documents,
    loadingDocs,
    uploading,
    uploadMsg,
    deletingId,
    error,
    languages,
    fetchDocuments,
    fetchLanguages,
    uploadFiles,
    deleteDocument,
    clearMessages,
  };

  return (
    <DocumentContext.Provider value={value}>
      {children}
    </DocumentContext.Provider>
  );
};
