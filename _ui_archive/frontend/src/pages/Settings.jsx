import React, { useState, useEffect } from 'react';
import api from '../api';

export default function Settings() {
   const [config, setConfig] = useState({
      API_BASE_URL: localStorage.getItem("API_BASE_URL") || "http://127.0.0.1:8000",
      MODEL_NAME: localStorage.getItem("MODEL_NAME") || "meta-llama/llama-3-8b-instruct:free",
      HF_TOKEN: localStorage.getItem("HF_TOKEN") || ""
   });
   
   const [saveStatus, setSaveStatus] = useState(null);

   const handleSave = async (e) => {
      e.preventDefault();
      
      // Save locally
      localStorage.setItem("API_BASE_URL", config.API_BASE_URL);
      localStorage.setItem("MODEL_NAME", config.MODEL_NAME);
      localStorage.setItem("HF_TOKEN", config.HF_TOKEN);
      
      try {
         // Push to backend
         await api.post(`/config`, config);
         setSaveStatus('success');
      } catch (err) {
         console.error(err);
         setSaveStatus('error');
      }
      
      setTimeout(() => setSaveStatus(null), 3000);
   };

   return (
      <div className="flex-1 p-8 fade-in">
         <h2 className="text-3xl font-bold text-white mb-6">Nexus Engine Configuration</h2>
         
         <form onSubmit={handleSave} className="bg-gray-800 p-8 rounded-xl max-w-xl space-y-6 shadow-2xl border border-gray-700 relative">
            {saveStatus === 'success' && <div className="absolute -top-4 right-4 bg-green-500 text-white font-bold py-1 px-4 rounded shadow-glow-green">Settings Deployed</div>}
            {saveStatus === 'error' && <div className="absolute -top-4 right-4 bg-red-500 text-white font-bold py-1 px-4 rounded shadow-glow-red">API Connection Failed</div>}
            
            <div className="group">
               <label className="block text-gray-300 font-semibold mb-2 group-focus-within:text-blue-400 transition-colors">API Base URL</label>
               <input type="text" className="w-full p-3 bg-gray-900 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow" 
                      value={config.API_BASE_URL} 
                      onChange={e => setConfig({...config, API_BASE_URL: e.target.value})} 
                      required />
               <p className="text-xs text-gray-500 mt-2">FastAPI instance (default: http://127.0.0.1:8000)</p>
            </div>
            <div className="group">
               <label className="block text-gray-300 font-semibold mb-2 group-focus-within:text-purple-400 transition-colors">Model Node Engine</label>
               <input type="text" className="w-full p-3 bg-gray-900 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-purple-500 transition-shadow" 
                      value={config.MODEL_NAME} 
                      onChange={e => setConfig({...config, MODEL_NAME: e.target.value})} 
                      required />
               <p className="text-xs text-gray-500 mt-2">HF or local inference endpoint target</p>
            </div>
            <div className="group">
               <label className="block text-gray-300 font-semibold mb-2 group-focus-within:text-green-400 transition-colors">Environment Token (HF_TOKEN)</label>
               <input type="password" placeholder="******************" className="w-full p-3 bg-gray-900 border border-gray-600 rounded text-white focus:outline-none focus:ring-2 focus:ring-green-500 shadow-inner" 
                      value={config.HF_TOKEN} 
                      onChange={e => setConfig({...config, HF_TOKEN: e.target.value})} />
               <p className="text-xs text-gray-500 mt-2">Keys are pushed to the backend node during runtime.</p>
            </div>
            <button type="submit" className="w-full py-4 text-lg bg-blue-600 hover:bg-blue-500 rounded font-bold text-white transition-all shadow-glow-blue hover:scale-[1.02]">
               Apply Variables & Deploy
            </button>
         </form>
      </div>
   );
}
