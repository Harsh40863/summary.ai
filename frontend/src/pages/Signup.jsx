import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../utils/api';
import '../styles/auth.css';

const Signup = () => {
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const getPasswordStrength = (pwd) => {
    if (!pwd) return { text: '', color: 'transparent', width: '0%' };
    if (pwd.length < 6) return { text: 'Weak', color: '#FF4D6D', width: '33%' };
    if (pwd.length <= 10) return { text: 'Fair', color: '#F59E0B', width: '66%' };
    return { text: 'Strong', color: '#10B981', width: '100%' };
  };

  const strength = getPasswordStrength(password);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await api.post('/auth/signup', { name, email, password });
      navigate('/login');
    } catch (err) {
      setError(err.response?.data?.detail || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-bg">
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
        <h1 className="auth-title">Join HackIndia</h1>
        <p className="auth-subtitle">Create your account and start building</p>
        
        <form onSubmit={handleSubmit} className="auth-form">
          <div className="input-group">
            <input type="text" placeholder="Full Name" value={name} onChange={e => setName(e.target.value)} required />
          </div>
          <div className="input-group">
            <input type="email" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} required />
          </div>
          <div className="input-group">
            <input type={showPassword ? "text" : "password"} placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
            <button type="button" className="eye-toggle" onClick={() => setShowPassword(!showPassword)}>
              {showPassword ? "Hide" : "Show"}
            </button>
          </div>
          {password && (
            <div className="password-strength-container">
              <div className="strength-bar" style={{ width: strength.width, backgroundColor: strength.color }}></div>
              <span style={{ color: strength.color, fontSize: '12px' }}>{strength.text}</span>
            </div>
          )}
          <div className="input-group">
            <input type="password" placeholder="Confirm Password" value={confirmPassword} onChange={e => setConfirmPassword(e.target.value)} required />
          </div>
          
          <button type="submit" className="auth-btn" disabled={loading}>
            {loading ? <div className="spinner"></div> : "Create Account"}
          </button>
          {error && <div className="auth-error">{error}</div>}
        </form>
        <div className="auth-footer">
          Already have an account? <Link to="/login">Sign in</Link>
        </div>
      </div>
    </div>
  );
};

export default Signup;
