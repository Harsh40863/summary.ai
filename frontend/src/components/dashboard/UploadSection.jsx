import React, { useState, useRef } from 'react';
import useDocuments from '../../hooks/useDocuments';

const UploadSection = () => {
  const { uploadFiles, uploading, uploadMsg, error } = useDocuments();
  const [selectedFiles, setSelectedFiles] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    setSelectedFiles(e.target.files);
  };

  const handleAction = async (e) => {
    e.preventDefault();
    if (selectedFiles && selectedFiles.length > 0) {
      const res = await uploadFiles(selectedFiles);
      if (res.success) {
        setSelectedFiles(null);
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
        }
      }
    } else {
      fileInputRef.current?.click();
    }
  };

  return (
    <div className="upload-section">
      <input
        type="file"
        multiple
        accept=".pdf,.doc,.docx,.txt"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: 'none' }}
      />
      <button
        className="btn-upload"
        onClick={handleAction}
        disabled={uploading}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          style={{ marginRight: '8px' }}
        >
          <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
        </svg>
        {uploading
          ? 'Uploading...'
          : selectedFiles && selectedFiles.length > 0
          ? `Upload ${selectedFiles.length} file(s)`
          : 'Upload Files'}
      </button>
      {uploadMsg && <div className="upload-success">{uploadMsg}</div>}
      {error && <div className="upload-error">{error}</div>}
    </div>
  );
};

export default React.memo(UploadSection);
