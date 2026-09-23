import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { DocumentProvider } from './context/DocumentContext';
import { SearchProvider } from './context/SearchContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoadingSpinner from './components/common/LoadingSpinner';

// Route-level code-splitting for instant initial page loads
const Home = lazy(() => import('./pages/Home'));
const Login = lazy(() => import('./pages/Login'));
const Signup = lazy(() => import('./pages/Signup'));
const Dashboard = lazy(() => import('./pages/Dashboard'));

function App() {
  return (
    <AuthProvider>
      <DocumentProvider>
        <SearchProvider>
          <BrowserRouter>
            <Suspense
              fallback={
                <div style={{
                  height: '100vh',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: '#07070F'
                }}>
                  <LoadingSpinner label="Loading..." />
                </div>
              }
            >
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/login" element={<Login />} />
                <Route path="/signup" element={<Signup />} />
                <Route
                  path="/dashboard"
                  element={
                    <ProtectedRoute>
                      <Dashboard />
                    </ProtectedRoute>
                  }
                />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </Suspense>
          </BrowserRouter>
        </SearchProvider>
      </DocumentProvider>
    </AuthProvider>
  );
}

export default App;
