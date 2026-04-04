import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import api from '../api';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

export default function Analytics() {
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
    const interval = setInterval(fetchHistory, 2500); 
    return () => clearInterval(interval);
  }, []);

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { position: 'top', labels: { color: 'white' } },
    },
    scales: {
      x: { ticks: { color: 'white' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
      y: { ticks: { color: 'white' }, grid: { color: 'rgba(255, 255, 255, 0.1)' } },
    }
  };

  const rewardData = {
    labels: history.map((h, i) => `Step ${i + 1}`),
    datasets: [{
      label: 'Performance Reward Matrix',
      data: history.map(h => h.reward),
      borderColor: 'rgb(59, 130, 246)',
      backgroundColor: 'rgba(59, 130, 246, 0.5)',
      tension: 0.3,
      fill: true
    }]
  };

  return (
    <div className="flex-1 p-8 overflow-auto fade-in">
      <h2 className="text-3xl font-bold text-white mb-6">Execution Analytics</h2>
      <div className="bg-gray-800 p-6 rounded-lg max-w-4xl border border-gray-700 shadow-xl h-[400px] mb-8 relative">
         {history.length > 0 ? (
           <Line options={chartOptions} data={rewardData} />
         ) : (
           <p className="text-gray-400 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">Run simulation matrix to render timeline</p>
         )}
      </div>

      {history.length > 0 && (
         <div className="mt-8 bg-gray-900 border border-gray-800 p-6 rounded-lg max-w-2xl">
            <h3 className="text-xl font-bold text-white mb-4">Meta Diagnostics</h3>
            <p className="text-gray-400 mb-2">Simulation steps logged: <span className="text-cyan-400 font-bold">{history.length}</span></p>
            <p className="text-gray-400 mb-2">Final/Current Reward: <span className="text-green-400 font-bold">{history[history.length - 1].reward.toFixed(2)}</span></p>
         </div>
      )}
    </div>
  );
}
