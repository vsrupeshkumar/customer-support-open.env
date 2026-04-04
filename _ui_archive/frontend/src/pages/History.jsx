import React, { useState, useEffect } from 'react';
import api from '../api';

export default function History() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await api.get(`/history`);
        setHistory(res.data);
      } catch (e) {
        console.error(e);
      }
    };
    fetchHistory();
    const interval = setInterval(fetchHistory, 3000); 
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex-1 p-8 overflow-auto fade-in">
      <h2 className="text-3xl font-bold text-white mb-6">Execution Log</h2>
      
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-4 shadow-xl h-full pb-[200px]">
        {history.length > 0 ? (
          <div className="space-y-4 max-h-[800px] overflow-y-auto custom-scrollbar">
            {history.slice().reverse().map((stepObj, idx) => (
              <div key={idx} className="bg-gray-800 p-4 rounded text-sm border-l-4 border-blue-500 hover:bg-gray-750 transition-colors">
                 <div className="flex justify-between items-center mb-2">
                    <span className="font-bold text-blue-400">Step {stepObj.step}</span>
                    <span className={`font-mono font-bold ${stepObj.reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      Reward: {stepObj.reward.toFixed(2)}
                    </span>
                 </div>
                 <pre className="text-gray-300 font-mono text-xs bg-gray-900 p-3 rounded overflow-x-auto">
                    {JSON.stringify(stepObj.action, null, 2)}
                 </pre>
                 {stepObj.done && <p className="mt-2 text-yellow-400 font-bold bg-yellow-900/40 p-2 rounded">ENVIRONMENT TERMINATED.</p>}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 italic p-6">Awaiting simulation execution... Boot matrix isolated.</p>
        )}
      </div>
    </div>
  );
}
