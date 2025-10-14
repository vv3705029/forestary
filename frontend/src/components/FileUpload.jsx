import React, { useState } from 'react';
import { satelliteAPI } from '../services/api';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [coordinates, setCoordinates] = useState({ lat: '', lon: '' });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [uploadMode, setUploadMode] = useState('file'); // 'file' or 'coordinates'

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      // Validate file type
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please select an image file');
        return;
      }
      // Validate file size (5MB limit)
      if (selectedFile.size > 5 * 1024 * 1024) {
        setError('File size must be less than 5MB');
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleCoordinateChange = (e) => {
    const { name, value } = e.target;
    setCoordinates(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const validateCoordinates = () => {
    const lat = parseFloat(coordinates.lat);
    const lon = parseFloat(coordinates.lon);
    
    if (isNaN(lat) || isNaN(lon)) {
      setError('Please enter valid numeric coordinates');
      return false;
    }
    
    if (lat < -90 || lat > 90) {
      setError('Latitude must be between -90 and 90');
      return false;
    }
    
    if (lon < -180 || lon > 180) {
      setError('Longitude must be between -180 and 180');
      return false;
    }
    
    return true;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;
      
      if (uploadMode === 'file') {
        if (!file) {
          setError('Please select a file');
          setLoading(false);
          return;
        }
        response = await satelliteAPI.uploadFile(file);
      } else {
        if (!validateCoordinates()) {
          setLoading(false);
          return;
        }
        response = await satelliteAPI.getSatelliteTile(coordinates.lat, coordinates.lon);
      }

      setResult(response);
    } catch (err) {
      setError(err.response?.data?.error || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setCoordinates({ lat: '', lon: '' });
    setResult(null);
    setError(null);
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
        Satellite Deforestation Prediction
      </h2>

      {/* Mode Selection */}
      <div className="mb-6">
        <div className="flex space-x-4 justify-center">
          <button
            onClick={() => setUploadMode('file')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              uploadMode === 'file'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Upload Image File
          </button>
          <button
            onClick={() => setUploadMode('coordinates')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              uploadMode === 'coordinates'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
            }`}
          >
            Use Coordinates
          </button>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {uploadMode === 'file' ? (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload Satellite Image
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-gray-400 transition-colors">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="hidden"
                id="file-upload"
              />
              <label
                htmlFor="file-upload"
                className="cursor-pointer flex flex-col items-center"
              >
                <svg
                  className="w-12 h-12 text-gray-400 mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                  />
                </svg>
                <span className="text-gray-600">
                  {file ? file.name : 'Click to select an image file'}
                </span>
                <span className="text-sm text-gray-500 mt-1">
                  PNG, JPG, JPEG up to 5MB
                </span>
              </label>
            </div>
            {file && (
              <div className="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-green-800 text-sm">
                  Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                </p>
              </div>
            )}
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Latitude
              </label>
              <input
                type="number"
                name="lat"
                value={coordinates.lat}
                onChange={handleCoordinateChange}
                placeholder="e.g., 40.7128"
                step="any"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Longitude
              </label>
              <input
                type="number"
                name="lon"
                value={coordinates.lon}
                onChange={handleCoordinateChange}
                placeholder="e.g., -74.0060"
                step="any"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required
              />
            </div>
          </div>
        )}

        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 text-sm">{error}</p>
          </div>
        )}

        <div className="flex space-x-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 bg-blue-600 text-white py-3 px-6 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? (
              <div className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </div>
            ) : (
              'Analyze Image'
            )}
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="px-6 py-3 border border-gray-300 text-gray-700 rounded-lg font-medium hover:bg-gray-50 transition-colors"
          >
            Reset
          </button>
        </div>
      </form>

      {result && (
        <div className="mt-8 p-6 bg-gray-50 rounded-lg">
          <h3 className="text-xl font-semibold text-gray-800 mb-4">Analysis Results</h3>
          <div className="space-y-4">
            <div className="p-4 bg-white rounded-lg border">
              <h4 className="font-medium text-gray-700 mb-2">Prediction Data:</h4>
              <pre className="text-sm text-gray-600 overflow-x-auto">
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
            {result.message && (
              <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                <p className="text-green-800 font-medium">{result.message}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUpload;
