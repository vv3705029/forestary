import axios from 'axios';

const API_BASE_URL = 'http://localhost:3000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
});

// API service for satellite tile prediction
export const satelliteAPI = {
  // Get satellite tile prediction by coordinates
  getSatelliteTile: async (lat, lon) => {
    try {
      const response = await api.get('/getSatelliteTile', {
        params: { lat, lon }
      });
      return response.data;
    } catch (error) {
      console.error('Error fetching satellite tile:', error);
      throw error;
    }
  },

  // Upload file for deforestation prediction
  uploadFile: async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await api.post('/deforestpredict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error uploading file:', error);
      throw error;
    }
  }
};

// API service for fire risk prediction
export const fireAPI = {
  // Predict fire risk based on location and date
  predictFireRisk: async (data) => {
    try {
      const response = await api.post('/fire-risk', data);
      return response.data;
    } catch (error) {
      console.error('Error predicting fire risk:', error);
      throw error;
    }
  },

  // Direct fire prediction (alternative endpoint)
  predictFire: async (data) => {
    try {
      const response = await api.post('/predictFire', data);
      return response.data;
    } catch (error) {
      console.error('Error predicting fire:', error);
      throw error;
    }
  }
};

export default api;