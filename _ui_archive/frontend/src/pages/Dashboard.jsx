import React, { useState, useEffect } from 'react';
import api from '../api';

export default function Dashboard() {
  const [metrics, setMetrics] = useState({ efficiency: 0, stability: 0, hazards_prevented: 0, total_reward: 0 });
  const [liveState, setLiveState] = useState(null);

  useEffect(() => {
    const fetchLive = async () => {
      try {
        const metricsRes = await api.get(`/metrics`);
        const stateRes = await api.get(`/live-state`);
        setMetrics(metricsRes.data);
        setLiveState(stateRes.data);
      } catch (e) {
        console.error("Dashboard error:", e);
      }
    };
    fetchLive();
    const interval = setInterval(fetchLive, 2500); 
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 p-8 overflow-auto fade-in">
      <h2 className="text-3xl font-bold text-white mb-6">Status Dashboard</h2>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="p-6 bg-gray-800 rounded-lg border border-gray-700 shadow-lg glow-blue transition-transform hover:scale-105">
           <p className="text-gray-400 font-semibold mb-2">Total Reward</p>
           <h3 className={`text-4xl font-bold ${metrics.total_reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
             {metrics.total_reward.toFixed(2)}
           </h3>
        </div>
        <div className="p-6 bg-gray-800 rounded-lg border border-gray-700 shadow-lg glow-green transition-transform hover:scale-105">
           <p className="text-gray-400 font-semibold mb-2">Efficiency</p>
           <h3 className="text-4xl text-green-400 font-bold">{(metrics.efficiency * 100).toFixed(1)}%</h3>
        </div>
        <div className="p-6 bg-gray-800 rounded-lg border border-gray-700 shadow-lg glow-purple transition-transform hover:scale-105">
           <p className="text-gray-400 font-semibold mb-2">Stability</p>
           <h3 className="text-4xl text-purple-400 font-bold">{(metrics.stability * 100).toFixed(1)}%</h3>
        </div>
        <div className="p-6 bg-gray-800 rounded-lg border border-gray-700 shadow-lg glow-pink transition-transform hover:scale-105">
           <p className="text-gray-400 font-semibold mb-2">Prevented Cascades</p>
           <h3 className="text-4xl text-pink-400 font-bold">{metrics.hazards_prevented}</h3>
        </div>
      </div>
      
      <h3 className="text-2xl font-bold text-white mb-4">Quick Overview</h3>
      <div className="grid grid-cols-3 gap-6">
         {!liveState && <p className="text-gray-500 italic">No environment data loaded...</p>}
         {liveState?.zones && Object.entries(liveState.zones).map(([k, v]) => (
            <div key={k} className="bg-gray-800 p-5 rounded-lg border border-gray-700 shadow-md hover:border-gray-500 transition-colors">
               <h4 className="text-xl text-white font-bold mb-3 border-b border-gray-700 pb-2">{k}</h4>
               <p className="text-md text-red-400 font-semibold">Fire: {v.fire}</p>
               <p className="text-md text-magenta-400 font-semibold">Patient: {v.patient}</p>
               <p className="text-md text-yellow-400 font-semibold">Traffic: {v.traffic}</p>
            </div>
         ))}
      </div>
    </div>
  );
}
