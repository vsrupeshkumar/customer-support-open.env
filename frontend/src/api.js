import axios from 'axios';

const api = axios.create({
  baseURL: localStorage.getItem('API_BASE_URL') || 'http://127.0.0.1:8000',
});

api.interceptors.request.use(config => {
  config.baseURL = localStorage.getItem('API_BASE_URL') || 'http://127.0.0.1:8000';
  return config;
});

export default api;
