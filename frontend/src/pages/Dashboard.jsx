import { useEffect } from 'react';
import Navbar from '../components/Navbar';
import { startKeepAlive, stopKeepAlive } from '../utils/api';
import useAuth from '../hooks/useAuth';
import useDocuments from '../hooks/useDocuments';
import UploadSection from '../components/dashboard/UploadSection';
import DocumentList from '../components/dashboard/DocumentList';
import QueryControls from '../components/dashboard/QueryControls';
import ResultSection from '../components/dashboard/ResultSection';
import '../styles/dashboard.css';

const Dashboard = () => {
  const { user } = useAuth();
  const { documents, fetchDocuments, fetchLanguages } = useDocuments();

  useEffect(() => {
    // Ping backend to prevent cold start idle sleep
    startKeepAlive();

    // Parallel data fetching with Promise.allSettled — zero waterfall delays
    Promise.allSettled([
      fetchDocuments(),
      fetchLanguages()
    ]);

    return () => {
      stopKeepAlive();
    };
  }, [fetchDocuments, fetchLanguages]);

  return (
    <div className="dashboard-wrapper">
      <div className="bg-animated">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="grid-overlay" />
      </div>

      <video
        className="dash-bg-video"
        src="/assets/bg-video.mp4"
        autoPlay
        loop
        muted
        playsInline
        preload="metadata"
        onError={(e) => {
          e.currentTarget.style.display = 'none';
        }}
      />
      <div className="dash-video-overlay" />

      <div className="dash-nav-container">
        <Navbar isLoggedIn={true} />
      </div>

      <div className="dash-main">
        {/* LEFT PANEL: Document Management */}
        <div className="left-panel">
          <div className="panel-header">My Documents ({documents.length})</div>
          <UploadSection />
          <DocumentList />
        </div>

        {/* RIGHT PANEL: AI Query & Results */}
        <div className="right-panel">
          <div className="welcome-header">
            Welcome back, <span className="welcome-name">{user?.name || 'Hacker'}</span>
          </div>
          <div className="welcome-sub">Explore your documents with AI power.</div>

          <QueryControls />
          <ResultSection />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
