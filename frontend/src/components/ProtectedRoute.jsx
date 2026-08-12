import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem('hackindia_token');
  
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  
  try {
    // Decode JWT payload (base64url)
    const payload = JSON.parse(atob(token.split('.')[1]));
    // Check if expired
    if (payload.exp && payload.exp * 1000 < Date.now()) {
      localStorage.removeItem('hackindia_token');
      return <Navigate to="/login" replace />;
    }
  } catch (e) {
    // If token is invalid/malformed
    localStorage.removeItem('hackindia_token');
    return <Navigate to="/login" replace />;
  }
  
  return children;
};

export default ProtectedRoute;
