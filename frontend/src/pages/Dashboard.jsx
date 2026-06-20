import { useState, useEffect, useRef } from 'react';
import Navbar from '../components/Navbar';
import api from '../utils/api';
import '../styles/dashboard.css';

const Dashboard = () => {
  const [name, setName] = useState('');
  const [prompt, setPrompt] = useState('');
  const [action, setAction] = useState('search');
  const [language, setLanguage] = useState('en');
  const [languages, setLanguages] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [files, setFiles] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('');
  const [documents, setDocuments] = useState([]);
  const [deletingId, setDeletingId] = useState(null);
  const [shake, setShake] = useState(false);
  const [queryError, setQueryError] = useState('');
  const [copiedStates, setCopiedStates] = useState({});
  const fileInputRef = useRef(null);
  
  const fetchDocuments = async () => {
    try {
      const res = await api.get('/api/documents');
      setDocuments(res.data.documents || []);
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  };

  useEffect(() => {
    const fetchUser = async () => {
      try {
        const res = await api.get('/auth/me');
        setName(res.data.name);
      } catch (err) {
        console.error(err);
      }
    };
    const fetchLanguages = async () => {
      try {
        const res = await api.get('/api/languages');
        setLanguages(res.data.languages || {});
      } catch (err) {
        console.error("Failed to fetch languages", err);
      }
    };
    fetchUser();
    fetchLanguages();
    fetchDocuments();
  }, []);

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!files || files.length === 0) return;
    setUploading(true);
    setUploadMsg('');
    setError('');
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }
    
    try {
      const res = await api.post('/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setUploadMsg(res.data.message || 'Files uploaded successfully!');
      setFiles(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
      fetchDocuments();
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleDeleteDocument = async (docId) => {
    setDeletingId(docId);
    try {
      await api.delete(`/api/documents/${docId}`);
      setTimeout(() => {
        setDocuments(prev => prev.filter(doc => doc.id !== docId));
        setDeletingId(null);
      }, 300);
    } catch (err) {
      console.error("Failed to delete document", err);
      setDeletingId(null);
      fetchDocuments();
    }
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setShake(true);
      setQueryError('Please enter a question first');
      setTimeout(() => {
        setShake(false);
        setQueryError('');
      }, 3000);
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const res = await api.post('/api/query', { 
        query: prompt,
        action: action,
        threshold: 0.35,
        translate_to: language
      });
      setResult(res.data);
    } catch (err) {
      setError(err.response?.data?.detail || 'Query failed');
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = (idx, text) => {
    navigator.clipboard.writeText(text);
    setCopiedStates(prev => ({ ...prev, [idx]: true }));
    setTimeout(() => {
      setCopiedStates(prev => ({ ...prev, [idx]: false }));
    }, 2000);
  };

  const formatMarkdown = (text) => {
    if (!text) return { __html: '' };
    let html = text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong style="color: #6C63FF;">$1</strong>');
    html = html.replace(/^[\s]*[\*\-]\s+(.*)$/gm, '<div style="margin-left: 16px; margin-bottom: 8px;">• $1</div>');
    html = html.replace(/\n/g, '<br/>');
    html = html.replace(/<br\/>(<div style="margin-left: 16px)/g, '$1');
    return { __html: html };
  };

  const formatBytes = (bytes) => {
    if (!bytes) return '';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div className="dashboard-wrapper">
      <div className="bg-animated">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="grid-overlay" />
      </div>
      <div className="dash-video-overlay" />
      
      <div className="dash-nav-container">
        <Navbar isLoggedIn={true} />
      </div>
      
      <div className="dash-main">
        {/* LEFT PANEL */}
        <div className="left-panel">
          <div className="panel-header">My Documents ({documents.length})</div>
          
          <div className="upload-section">
            <input 
              type="file" 
              multiple 
              accept=".pdf,.doc,.docx,.txt"
              ref={fileInputRef}
              onChange={e => {
                setFiles(e.target.files);
              }}
              style={{ display: 'none' }}
            />
            <button 
              className="btn-upload"
              onClick={(e) => {
                if (files && files.length > 0) {
                  handleUpload(e);
                } else {
                  fileInputRef.current?.click();
                }
              }}
              disabled={uploading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '8px' }}>
                <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"></path>
              </svg>
              {uploading ? 'Uploading...' : (files && files.length > 0 ? `Upload ${files.length} file(s)` : 'Upload Files')}
            </button>
            {uploadMsg && <div className="upload-success">{uploadMsg}</div>}
            {error && <div className="upload-error">{error}</div>}
          </div>

          <div className="doc-list">
            {documents.length === 0 ? (
              <div className="empty-docs">No documents yet. Upload one above.</div>
            ) : (
              documents.map((doc, idx) => (
                <div key={idx} className="doc-item">
                  <div className="doc-filename">
                    📄 {doc.name} {doc.content_length && <span className="doc-size">({formatBytes(doc.content_length)})</span>}
                  </div>
                  <button 
                    className={`btn-delete ${deletingId === doc.id ? 'deleting' : ''}`} 
                    onClick={() => handleDeleteDocument(doc.id)}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                    </svg>
                  </button>
                </div>
              ))
            )}
          </div>
        </div>

        {/* RIGHT PANEL */}
        <div className="right-panel">
          <div className="welcome-header">Welcome back, <span className="welcome-name">{name || 'Hacker'}</span></div>
          <div className="welcome-sub">Explore your documents with AI power.</div>

          <div className="dropdowns-row">
            <select className="dash-select" value={action} onChange={e => setAction(e.target.value)}>
              <option value="search">Search</option>
              <option value="explore">Explore (Web Enhanced)</option>
              <option value="think">Think (AI Insights)</option>
              <option value="ppt">Generate PPT</option>
            </select>
            
            <select className="dash-select" value={language} onChange={e => setLanguage(e.target.value)}>
              {Object.entries(languages).map(([code, langName]) => (
                <option key={code} value={code}>{langName}</option>
              ))}
              {Object.keys(languages).length === 0 && <option value="en">English</option>}
            </select>
          </div>

          <textarea
            className={`dash-textarea ${shake ? 'shake' : ''}`}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Ask anything about your documents..."
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
              </div>
            ) : 'Generate Magic'}
          </button>
          
          {result && (
            <div className="result-container result-card">
              <div className="result-label">🔍 Found {result.results?.length || 0} relevant documents</div>
              
              {result.results?.map((item, idx) => {
                const textToCopy = item.summary || item.refined_insight || (typeof item.web_content === 'string' ? item.web_content : JSON.stringify(item.web_content));
                
                return (
                  <div key={idx} className="result-card-inner">
                    <div className="result-card-header">
                      <div className="result-doc-name">📄 {item.document_name}</div>
                      <button className="btn-copy" onClick={() => handleCopy(idx, textToCopy)}>
                        {copiedStates[idx] ? <span style={{color: '#10B981'}}>Copied ✓</span> : 'Copy'}
                      </button>
                    </div>
                    <hr className="result-hr" />
                    
                    <div className="result-text-area">
                      {item.summary && (
                        <div dangerouslySetInnerHTML={formatMarkdown(item.summary)}></div>
                      )}
                      {item.refined_insight && (
                        <div dangerouslySetInnerHTML={formatMarkdown(item.refined_insight)}></div>
                      )}
                      {item.web_content && (
                        <div>
                          {typeof item.web_content === 'object' ? JSON.stringify(item.web_content, null, 2) : <div dangerouslySetInnerHTML={formatMarkdown(item.web_content)}></div>}
                        </div>
                      )}
                      {item.ppt_path && (
                        <div style={{ color: '#10B981', marginTop: '8px', fontWeight: '500' }}>✨ PPT Saved: {item.file_name}</div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
