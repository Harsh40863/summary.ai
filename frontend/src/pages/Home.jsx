import { Link } from 'react-router-dom';
import Navbar from '../components/Navbar';
import '../styles/home.css';

const Home = () => {
  return (
    <div className="home-wrapper">
      <div className="bg-animated">
        <div className="orb orb-1" />
        <div className="orb orb-2" />
        <div className="orb orb-3" />
        <div className="grid-overlay" />
      </div>
      <div className="video-overlay" />
      <Navbar />
      <div className="hero-content">
        <h1 className="hero-title">
          Hack<span style={{ color: '#6C63FF' }}>India</span>
        </h1>
        <div className="hero-tagline">Build. Break. Win.</div>
        <p className="hero-desc">
          AI-powered tools for the next generation of hackers
        </p>
        <div className="hero-actions">
          <Link to="/signup" className="btn-get-started">Get Started</Link>
          <Link to="/login" className="btn-sign-in">Sign In</Link>
        </div>
      </div>
    </div>
  );
};

export default Home;
