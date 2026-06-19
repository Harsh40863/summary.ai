import { Link, useNavigate } from 'react-router-dom';

const Navbar = ({ isLoggedIn }) => {
  const navigate = useNavigate();
  const token = localStorage.getItem('hackindia_token');
  const isAuth = isLoggedIn || !!token;

  const handleLogout = () => {
    localStorage.removeItem('hackindia_token');
    navigate('/');
  };

  return (
    <nav style={{
      position: 'absolute',
      top: 0,
      width: '100%',
      zIndex: 10,
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '20px 40px',
      background: 'linear-gradient(to bottom, rgba(0,0,0,0.5), transparent)',
      fontFamily: "'Space Grotesk', sans-serif"
    }}>
      <div style={{ fontSize: '24px', fontWeight: 'bold' }}>
        <Link to="/" style={{ color: 'white', textDecoration: 'none' }}>
          Hack<span style={{ color: '#6C63FF' }}>India</span>
        </Link>
      </div>
      <div>
        {!isAuth ? (
          <>
            <Link to="/login" style={{ color: 'white', marginRight: '20px', textDecoration: 'none' }}>Sign In</Link>
            <Link to="/signup" style={{ color: 'white', textDecoration: 'none' }}>Sign Up</Link>
          </>
        ) : (
          <>
            <Link to="/dashboard" style={{ color: 'white', marginRight: '20px', textDecoration: 'none' }}>Dashboard</Link>
            <button onClick={handleLogout} style={{
              background: 'transparent', border: 'none', color: 'white', cursor: 'pointer', fontFamily: "'Space Grotesk', sans-serif", fontSize: '16px'
            }}>Logout</button>
          </>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
