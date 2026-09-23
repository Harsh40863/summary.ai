import React from 'react';

const LoadingSpinner = ({ label = 'Loading...' }) => {
  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '200px',
      color: '#A0A5B5',
      fontFamily: "'Space Grotesk', sans-serif",
      gap: '14px'
    }}>
      <div style={{
        width: '32px',
        height: '32px',
        border: '3px solid rgba(108, 99, 255, 0.2)',
        borderTop: '3px solid #6C63FF',
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite'
      }} />
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
      <span style={{ fontSize: '14px', letterSpacing: '0.5px' }}>{label}</span>
    </div>
  );
};

export default React.memo(LoadingSpinner);
