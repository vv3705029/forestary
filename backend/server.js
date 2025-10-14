const express = require('express');
const multer = require('multer');
const axios = require('axios');
const FormData = require('form-data');
const cors = require('cors');
const dayjs = require('dayjs');

const app = express();
const port = process.env.PORT || 3000;

// CORS configuration
app.use(cors({
  origin: ['http://localhost:5173', 'http://localhost:5174', 'http://localhost:3000'],
  credentials: true,
}));

// Middleware
app.use(express.json());

// Multer config: memory storage with file size limit (e.g., 5MB)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 }, // 5 MB limit
});

// Root route - health check
app.get('/', (req, res) => {
  res.send('Planetary Environmental Monitoring API is running');
});

// Convert lat/lon to tile x/y at zoom level
function deg2tile(lat, lon, zoom) {
  const latRad = (lat * Math.PI) / 180;
  const n = Math.pow(2, zoom);
  const x = Math.floor(((lon + 180) / 360) * n);
  const y = Math.floor(
    ((1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2) * n
  );
  return { x, y };
}

// GET /getSatelliteTile?lat=...&lon=...
app.get('/getSatelliteTile', async (req, res) => {
  const { lat, lon } = req.query;

  if (!lat || !lon) {
    return res.status(400).json({ error: 'Latitude and longitude are required.' });
  }

  const zoom = 14; // You can configure zoom level as needed
  const { x, y } = deg2tile(parseFloat(lat), parseFloat(lon), zoom);

  // ESRI Satellite tile URL
  const tileUrl = `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${zoom}/${y}/${x}`;

  try {
    // Fetch tile image as binary buffer
    const tileResponse = await axios.get(tileUrl, { responseType: 'arraybuffer' });
    const tileBuffer = Buffer.from(tileResponse.data);

    // Wrap buffer in FormData for upload
    const formData = new FormData();
    formData.append('file', tileBuffer, {
      filename: `tile_${lat}_${lon}.jpg`,
      contentType: 'image/jpeg',
    });

    // Forward image to /deforestpredict route (on this same server)
    const predictionResponse = await axios.post(
      `http://localhost:${port}/deforestpredict`,
      formData,
      {
        headers: formData.getHeaders(),
        maxContentLength: Infinity,
        maxBodyLength: Infinity,
        timeout: 10000, // 10s timeout, adjust as needed
      }
    );

    res.json({
      message: 'Prediction successful',
      data: predictionResponse.data,
    });
  } catch (error) {
    console.error('Error in /getSatelliteTile:', error.message);
    if (error.response) {
      console.error('Response data:', error.response.data);
    }
    res.status(500).json({ error: 'Failed to fetch satellite tile or get prediction.' });
  }
});

// POST /deforestpredict - accepts image file, forwards to external ML model API, returns prediction
app.post('/deforestpredict', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No image file uploaded.' });
  }

  console.log(`Received file for prediction: ${req.file.originalname} (${req.file.size} bytes)`);

  try {
    // Prepare FormData to send to external ML model
    const formData = new FormData();
    formData.append('file', req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // Replace this URL with your actual ML model endpoint
    const mlModelUrl = process.env.ML_MODEL_URL || 'http://localhost:8000/predict';

    const modelResponse = await axios.post(mlModelUrl, formData, {
      headers: formData.getHeaders(),
      maxContentLength: Infinity,
      maxBodyLength: Infinity,
      timeout: 20000, // 20s timeout, adjust as needed
    });

    // Forward the ML model JSON response as-is
    res.json(modelResponse.data);
  } catch (error) {
    console.error('Error forwarding file to ML model:', error.message);
    if (error.response) {
      console.error('ML model response error data:', error.response.data);
    }
    res.status(500).json({ error: 'Prediction failed. Please try again later.' });
  }
});

// Weather API configuration
const WEATHER_API_KEY = 'a46320d5c05842c7968132657251110';
const WEATHER_API_URL = 'https://api.weatherapi.com/v1/history.json';

// Fallback weather data for when API fails
const getFallbackWeather = (lat, lon, date) => {
  const dateObj = new Date(date);
  const month = dateObj.getMonth() + 1; // 1-12
  const isSummer = month >= 6 && month <= 8;
  const isWinter = month >= 12 || month <= 2;

  let baseTemp = 20;
  let baseHumidity = 50;
  let baseWind = 10;
  let basePrecip = 0;

  if (isSummer) {
    baseTemp = 25 + Math.random() * 10; // 25-35°C
    baseHumidity = 40 + Math.random() * 20; // 40-60%
    baseWind = 8 + Math.random() * 8; // 8-16 km/h
    basePrecip = Math.random() < 0.3 ? Math.random() * 5 : 0; // 30% chance of rain
  } else if (isWinter) {
    baseTemp = 5 + Math.random() * 10; // 5-15°C
    baseHumidity = 60 + Math.random() * 20; // 60-80%
    baseWind = 12 + Math.random() * 10; // 12-22 km/h
    basePrecip = Math.random() < 0.4 ? Math.random() * 3 : 0; // 40% chance of rain
  } else {
    baseTemp = 15 + Math.random() * 10; // 15-25°C
    baseHumidity = 50 + Math.random() * 20; // 50-70%
    baseWind = 8 + Math.random() * 8; // 8-16 km/h
    basePrecip = Math.random() < 0.2 ? Math.random() * 2 : 0; // 20% chance of rain
  }

  return {
    avgtemp_c: Math.round(baseTemp * 10) / 10,
    total_precip_mm: Math.round(basePrecip * 10) / 10,
    avg_humidity: Math.round(baseHumidity),
    pressure_in: 29.92 + (Math.random() - 0.5) * 2, // 28.92-30.92
    wind_kph: Math.round(baseWind * 10) / 10,
  };
};

// Fetch historical weather data for given lat, lon, and date
async function fetchWeather(lat, lon, date) {
  const dateStr = dayjs(date).format('YYYY-MM-DD');

  try {
    console.log(WEATHER_API_URL)
    console.log(`${lat},${lon}`)
    console.log(dateStr)
    const response = await axios.get(WEATHER_API_URL + `?key=${WEATHER_API_KEY}&q=${lat},${lon}&dt=${dateStr}`);

    console.log(WEATHER_API_URL + `?key=${WEATHER_API_KEY}&q=${lat},${lon}&dt=${dateStr}`)
    const data = response.data;
    const day = data.forecast.forecastday[0].day;

    return {
      avgtemp_c: day.avgtemp_c,
      total_precip_mm: day.totalprecip_mm,
      avg_humidity: day.avghumidity,
      pressure_in: data.current.pressure_in || 29.92,  // approximate fallback
      wind_kph: day.maxwind_kph,
    };
  } catch (error) {
    console.error('Weather API error:', error.message);
    console.log('Using fallback weather data for location:', lat, lon, 'date:', date);
    // Return realistic fallback weather data
    return getFallbackWeather(lat, lon, date);
  }
}

// POST /fire-risk - fire risk prediction with weather data
app.post('/fire-risk', async (req, res) => {
  try {
    const { latitude, longitude, date } = req.body;

    if (!latitude || !longitude || !date) {
      return res.status(400).json({ error: 'latitude, longitude, and date are required' });
    }

    // Fetch weather data
    const weather = await fetchWeather(latitude, longitude, date);

    // Compose payload for FastAPI /predict endpoint
    const dateObj = dayjs(date);

    const payload = {
      avgtemp_c: weather.avgtemp_c,
      total_precip_mm: weather.total_precip_mm,
      avg_humidity: weather.avg_humidity,
      pressure_in: weather.pressure_in,
      wind_kph: weather.wind_kph,
      acq_date: dateObj.format('YYYY-MM-DD'),
    };

    // Send to FastAPI /predict endpoint (replace with actual endpoint)
    const predictionResponse = await axios.post('http://localhost:8000/predict', payload);
    res.json(predictionResponse.data);
  } catch (error) {
    console.error('Error during fire risk prediction:', error.message);
    res.status(500).json({ error: 'Failed to predict fire risk. Please try again later.' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
