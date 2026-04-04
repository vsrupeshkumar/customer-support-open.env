import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, useLocation } from 'react-router-dom';

import Dashboard from './pages/Dashboard';
import LiveFeed from './pages/LiveFeed';
import Resources from './pages/Resources';
import History from './pages/History';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import './index.css';

const Sidebar = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const tabs = [
    { path: '/dashboard', label: 'Dashboard', icon: '📊' },
    { path: '/live-feed', label: 'Live Feed', icon: '🔴' },
    { path: '/resources', label: 'Resources', icon: '🚒' },
    { path: '/history', label: 'History', icon: '📜' },
    { path: '/analytics', label: 'Analytics', icon: '📈' },
    { path: '/settings', label: 'Settings', icon: '⚙️' }
  ];

  return (
    <aside className="w-72 bg-gray-900 border-r border-gray-800 flex flex-col p-6 space-y-6 shadow-2xl relative z-10">
      <h1 className="text-2xl font-black text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600 tracking-widest uppercase mb-8 text-center drop-shadow-[0_0_10px_rgba(56,189,248,0.8)]">OpenEnv Nexus</h1>
      <nav className="flex flex-col space-y-3 flex-1 px-2">
      {tabs.map((tab) => {
        const isActive = location.pathname.includes(tab.path) || (location.pathname === '/' && tab.path === '/dashboard');
        return (
          <button
            key={tab.path}
            onClick={() => navigate(tab.path)}
            className={`text-left p-4 rounded-xl transition-all duration-300 font-bold flex items-center shadow-md ${
              isActive 
                ? "bg-blue-600 border border-blue-500 text-white shadow-[0_0_15px_rgba(59,130,246,0.5)] scale-105 z-20" 
                : "bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200 border border-transparent hover:border-gray-600 hover:scale-[1.02]"
            }`}
          >
            <span className="mr-4 text-xl drop-shadow-md">{tab.icon}</span> 
            {tab.label}
          </button>
        )
      })}
      </nav>
      
      <div className="mt-auto pt-6 text-center text-xs text-gray-600 border-t border-gray-800 font-mono tracking-wide">
        v1.0.0 (Hackathon Build)
      </div>
    </aside>
  );
};

export default function App() {
  return (
    <Router>
      <div className="flex h-screen bg-[#050510] text-gray-200 font-sans overflow-hidden">
        <Sidebar />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/live-feed" element={<LiveFeed />} />
          <Route path="/resources" element={<Resources />} />
          <Route path="/history" element={<History />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<div className="flex-1 flex items-center justify-center text-red-500 font-black text-4xl animate-pulse tracking-widest drop-shadow-[0_0_20px_rgba(239,68,68,0.8)]">404 MODULE OFFLINE</div>} />
        </Routes>
      </div>
    </Router>
  );
}
