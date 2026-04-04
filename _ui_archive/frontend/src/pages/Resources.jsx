import React, { useState, useEffect } from 'react';
import api from '../api';

export default function Resources() {
  const [state, setState] = useState(null);

  useEffect(() => {
    const fetchRes = async () => {
      try {
        const res = await api.get(`/state`);
        setState(res.data.observation);
      } catch (e) {
        console.error(e);
      }
    };
    fetchRes();
    const interval = setInterval(fetchRes, 3000); 
    return () => clearInterval(interval);
  }, []);

  if (!state) return <div className="p-8 text-gray-500">Loading resources...</div>;

  return (
    <div className="flex-1 p-8 overflow-auto fade-in">
      <h2 className="text-3xl font-bold text-white mb-6">Resource Allocation Network</h2>
      
      <div className="grid grid-cols-2 gap-8">
        <div className="bg-gray-800 p-8 rounded-lg shadow-lg border-2 border-green-500/30">
          <h3 className="text-2xl font-bold text-green-400 mb-6 tracking-wide">IDLE (BASE)</h3>
          <div className="space-y-4">
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Fire Units</span>
                <span className="text-white font-mono text-xl">{state.idle_resources.fire_units}</span>
             </div>
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Ambulance</span>
                <span className="text-white font-mono text-xl">{state.idle_resources.ambulances}</span>
             </div>
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Police Units</span>
                <span className="text-white font-mono text-xl">{state.idle_resources.police}</span>
             </div>
          </div>
        </div>

        <div className="bg-gray-800 p-8 rounded-lg shadow-lg border-2 border-red-500/30">
          <h3 className="text-2xl font-bold text-red-500 mb-6 tracking-wide">ACTIVE DEPLOYED</h3>
          <div className="space-y-4">
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Fire Units</span>
                <span className="text-white font-mono text-xl">{state.busy_resources.fire_units}</span>
             </div>
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Ambulance</span>
                <span className="text-white font-mono text-xl">{state.busy_resources.ambulances}</span>
             </div>
             <div className="flex justify-between border-b border-gray-700 pb-2">
                <span className="text-gray-300 font-semibold">Police Units</span>
                <span className="text-white font-mono text-xl">{state.busy_resources.police}</span>
             </div>
          </div>
        </div>
      </div>
    </div>
  );
}
