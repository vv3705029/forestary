# Planetary Frontend

A React-based frontend application for satellite deforestation prediction analysis.

## Features

- **File Upload**: Upload satellite images directly for deforestation analysis
- **Coordinate Input**: Enter latitude and longitude coordinates to fetch satellite tiles
- **Real-time Analysis**: Get instant predictions from the ML model
- **Responsive Design**: Modern, mobile-friendly interface built with Tailwind CSS

## Getting Started

### Prerequisites

- Node.js (v16 or higher)
- Backend server running on port 3000

### Installation

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

3. Open your browser and navigate to `http://localhost:5173`

## Usage

### Upload Image File
1. Click "Upload Image File" mode
2. Select an image file (PNG, JPG, JPEG up to 5MB)
3. Click "Analyze Image" to get deforestation prediction

### Use Coordinates
1. Click "Use Coordinates" mode
2. Enter latitude and longitude values
3. Click "Analyze Image" to fetch satellite tile and get prediction

## API Integration

The frontend communicates with the backend API endpoints:

- `GET /getSatelliteTile?lat={lat}&lon={lon}` - Fetch satellite tile by coordinates
- `POST /deforestpredict` - Upload file for deforestation prediction

## Technologies Used

- React 19
- Vite
- Tailwind CSS
- Axios
- React Router DOM

## Project Structure

```
src/
├── components/
│   ├── FileUpload.jsx    # Main file upload component
│   └── Navbar.jsx        # Navigation component
├── services/
│   └── api.js            # API service layer
├── App.jsx               # Main application component
└── main.jsx              # Application entry point
```