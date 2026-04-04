import React, { useState, useEffect, useRef } from 'react';
import api from '../api';

export default function LiveFeed() {
  const [liveState, setLiveState] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const isRunningRef = useRef(isRunning);
  
  useEffect(() => {
    isRunningRef.current = isRunning;
  }, [isRunning]);

  const loadState = async () => {
    try {
      const stateRes = await api.get(`/live-state`);
      setLiveState(stateRes.data);
    } catch(e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadState();
    const interval = setInterval(() => {
      if(!isRunningRef.current) {
        loadState();
      }
    }, 2500); 
    return () => clearInterval(interval);
  }, []);

  const handleStartSim = async () => {
    await api.post(`/reset?task_id=3`);
    await loadState();
  };

  const handleStep = async () => {
    try {
      const stateRes = await api.get(`/state`);
      const obs = stateRes.data.observation;
      const agentRes = await api.post(`/agent_action?use_llm=false`, obs);
      await api.post(`/step`, agentRes.data.action); 
      await loadState();
    } catch(e) {
      console.error(e);
      setIsRunning(false);
    }
  }

  const toggleAuto = () => {
    setIsRunning(!isRunning);
  }

  useEffect(() => {
    let timer;
    if (isRunning) {
      timer = setInterval(() => {
        handleStep();
      }, 1500);
    }
    return () => clearInterval(timer);
  }, [isRunning]);

  return (
    <div className="flex-1 p-8 overflow-auto fade-in">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-3xl font-bold text-white flex items-center">
            Live Feed 
            {isRunning && <span className="ml-4 text-xs bg-red-600 text-white font-bold py-1 px-3 rounded-full animate-pulse">LIVE</span>}
        </h2>
      </div>

      <div className="flex space-x-4 mb-10 border-b border-gray-800 pb-6">
        <button onClick={handleStartSim} className="px-6 py-3 font-semibold bg-green-600 hover:bg-green-500 text-white rounded shadow-glow-green transition">
            🔄 Reset Simulation
        </button>
        <button onClick={handleStep} className="px-6 py-3 font-semibold bg-blue-600 hover:bg-blue-500 text-white rounded shadow-glow-blue transition">
            ⏭ Next Step
        </button>
        <button onClick={toggleAuto} className={`px-6 py-3 font-semibold text-white rounded transition ${isRunning ? 'bg-red-600 hover:bg-red-500 shadow-glow-red' : 'bg-purple-600 hover:bg-purple-500 shadow-glow-purple'}`}>
            {isRunning ? '⏹ Stop Auto Execute' : '▶ Start Auto Execute'}
        </button>
      </div>

      {liveState && (
         <div>
            <div className="flex justify-between bg-gray-800 p-4 rounded-lg border border-gray-700 mb-6">
               <div className="text-gray-300"><span className="text-blue-400 font-bold mr-2">STEP:</span> {liveState.step}/{liveState.max_steps}</div>
               <div className="text-gray-300"><span className="text-yellow-400 font-bold mr-2">WEATHER:</span> {liveState.weather.toUpperCase()}</div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {Object.entries(liveState.zones).map(([k, v]) => (
                    <div key={k} className="bg-gray-900 border border-gray-700 rounded-xl p-6 shadow-xl relative overflow-hidden">
                        <div className={`absolute top-0 left-0 w-full h-1 ${v.consecutive_failures > 0 ? 'bg-red-500' : 'bg-green-500'}`}></div>
                        <h4 className="text-2xl font-black text-white mb-4 tracking-wider">{k.toUpperCase()}</h4>
                        
                        <div className="space-y-3">
                            <div className="flex justify-between items-center bg-gray-800 p-3 rounded text-sm">
                                <span className="font-semibold text-red-400 w-1/3">FIRE</span> 
                                <span className="text-gray-200 font-mono uppercase">{v.fire}</span>
                            </div>
                            <div className="flex justify-between items-center bg-gray-800 p-3 rounded text-sm">
                                <span className="font-semibold text-pink-400 w-1/3">PATIENT</span> 
                                <span className="text-gray-200 font-mono uppercase">{v.patient}</span>
                            </div>
                            <div className="flex justify-between items-center bg-gray-800 p-3 rounded text-sm">
                                <span className="font-semibold text-yellow-400 w-1/3">TRAFFIC</span> 
                                <span className="text-gray-200 font-mono uppercase">{v.traffic}</span>
                            </div>
                            <div className="flex justify-between items-center bg-gray-800 p-3 rounded text-sm border border-gray-700">
                                <span className="font-semibold text-orange-400 w-1/3">RISK Lvl</span> 
                                <span className="text-white font-mono">{v.consecutive_failures} failures</span>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
         </div>
      )}
      {!liveState && <p className="text-gray-500">Connecting to OpenEnv node...</p>}
    </div>
  );
}
