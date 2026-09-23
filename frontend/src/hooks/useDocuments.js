import { useContext } from 'react';
import { DocumentContext } from '../context/DocumentContext';

export function useDocuments() {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within a DocumentProvider');
  }
  return context;
}

export default useDocuments;
