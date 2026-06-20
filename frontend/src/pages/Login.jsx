import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import '../styles/auth.css';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const response = await api.post('/auth/login', { email, password });
      localStorage.setItem('hackindia_token', response.data.access_token);
      navigate('/dashboard');
    } catch (err) {
      setError(err.response?.data?.detail || 'Login failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-bg">
      <div className="bg-animated">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="grid-overlay" />
      </div>
      <video
        className="auth-bg-video"
        src="/assets/auth-bg-video.mp4"
        autoPlay
        loop
        muted
        playsInline
      />
      <div className="auth-video-overlay" />
      <div className="auth-card">
        <div className="auth-logo">
          Hack<span>India</span>
        </div>
        <h1 className="auth-title">Welcome back</h1>
        <p className="auth-subtitle">Sign in to your HackIndia account</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <svg className="input-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
            <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} required />
          </div>
          <div className="input-group">
            <input type={showPassword ? "text" : "password"} placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
            <button type="button" className="eye-toggle" onClick={() => setShowPassword(!showPassword)}>
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          <div className="forgot-password">
            <a href="#">Forgot password?</a>
          </div>
          <button type="submit" className="auth-btn" disabled={loading}>
            {loading ? <div className="spinner"></div> : "Sign In"}
          </button>
          {error && <div className="auth-error">{error}</div>}
        </form>
        <div className="auth-footer">
          Don't have an account? <Link to="/signup">Create one</Link>
        </div>
      </div>
    </div>
  );
};

export default Login;
