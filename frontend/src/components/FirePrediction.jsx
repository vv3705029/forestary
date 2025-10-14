import React, { useState } from 'react';
import { fireAPI } from '../services/api';

// Fire Risk Level Component
const FireRiskLevel = ({ riskScore }) => {
  const getRiskLevel = (score) => {
    if (score >= 0.7) return { level: 'HIGH', color: '#ff4757', bgColor: 'rgba(255, 71, 87, 0.1)' };
    if (score >= 0.4) return { level: 'MODERATE', color: '#facc15', bgColor: 'rgba(250, 204, 21, 0.1)' };
    return { level: 'LOW', color: '#4ade80', bgColor: 'rgba(74, 222, 128, 0.1)' };
  };

  const risk = getRiskLevel(riskScore);
  const percentage = Math.round(riskScore * 100);

  return (
    <div style={{
      background: 'linear-gradient(135deg, #000000 0%, #111111 50%, #000000 100%)',
      borderRadius: '20px',
      border: `2px solid ${risk.color}`,
      padding: 'clamp(2rem, 5vw, 3rem)',
      marginBottom: '2rem',
      position: 'relative',
      overflow: 'hidden',
      boxShadow: `0 0 30px ${risk.color}30`
    }}>
      {/* Animated Background */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
          radial-gradient(circle at 20% 80%, ${risk.bgColor} 0%, transparent 50%),
          radial-gradient(circle at 80% 20%, ${risk.bgColor} 0%, transparent 50%)
        `,
        animation: 'pulse 4s ease-in-out infinite'
      }}></div>
      
      <div style={{ position: 'relative', zIndex: 2 }}>
        <h3 style={{
          fontSize: 'clamp(2rem, 5vw, 3rem)',
          fontWeight: '800',
          color: '#ffffff',
          marginBottom: '1rem',
          textAlign: 'center',
          textShadow: `0 0 30px ${risk.color}50`,
          letterSpacing: '-0.02em'
        }}>
          🔥 FIRE RISK ASSESSMENT
        </h3>
        
        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '15px',
          border: '1px solid rgba(255,255,255,0.2)',
          padding: '1.5rem',
          marginBottom: '2rem',
          textAlign: 'center',
          backdropFilter: 'blur(10px)'
        }}>
          <div style={{
            fontSize: 'clamp(1.2rem, 3vw, 1.5rem)',
            color: '#ffffff',
            fontWeight: '600',
            marginBottom: '0.5rem'
          }}>
            🎯 Wildfire Risk Analysis Complete
          </div>
          <div style={{
            fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
            color: '#cccccc'
          }}>
            AI-powered assessment based on weather conditions and historical data
          </div>
        </div>

        {/* Risk Score Display */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 'clamp(1rem, 4vw, 2rem)',
          flexWrap: 'wrap'
        }}>
          <div style={{ 
            position: 'relative', 
            width: 'clamp(120px, 25vw, 160px)', 
            height: 'clamp(120px, 25vw, 160px)',
            background: '#000000',
            borderRadius: '50%',
            padding: 'clamp(10px, 2vw, 15px)',
            border: `2px solid ${risk.color}`
          }}>
            <svg viewBox="0 0 100 100" width="100%" height="100%" style={{ transform: 'rotate(-90deg)' }}>
              {/* Background circle */}
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="none"
                stroke="#ffffff"
                strokeWidth="8"
              />
              {/* Risk arc */}
              <circle
                cx="50"
                cy="50"
                r="40"
                fill="none"
                stroke={risk.color}
                strokeWidth="12"
                strokeDasharray={251.2}
                strokeDashoffset={251.2 - (riskScore * 251.2)}
                strokeLinecap="butt"
                style={{ 
                  transition: 'stroke-dashoffset 1s ease-out',
                  filter: `drop-shadow(0 0 8px ${risk.color}40)`
                }}
              />
            </svg>
            <div style={{ 
              position: 'absolute', 
              top: '50%', 
              left: '50%', 
              transform: 'translate(-50%, -50%)',
              textAlign: 'center'
            }}>
              <div style={{ 
                fontSize: 'clamp(1.5rem, 4vw, 2rem)', 
                fontWeight: '700',
                color: risk.color,
                marginBottom: '0.25rem',
                textShadow: `0 0 20px ${risk.color}60`
              }}>
                {percentage}%
              </div>
              <div style={{ 
                fontSize: 'clamp(0.7rem, 2vw, 0.8rem)', 
                color: '#ffffff',
                fontWeight: '500'
              }}>
                Risk: {risk.level}
              </div>
            </div>
          </div>
          
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: 'clamp(0.75rem, 2vw, 1rem)',
            minWidth: 'clamp(200px, 40vw, 250px)'
          }}>
            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.75rem',
              padding: 'clamp(0.75rem, 2vw, 1rem)',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '10px',
              borderLeft: `5px solid ${risk.color}`,
              boxShadow: `0 0 15px ${risk.color}1a`
            }}>
              <div style={{ 
                fontSize: 'clamp(1.2rem, 3vw, 1.5rem)', 
                fontWeight: '700',
                color: risk.color 
              }}>
                {percentage}%
              </div>
              <div style={{ fontSize: 'clamp(0.9rem, 2.5vw, 1rem)', color: '#ffffff' }}>
                Fire Risk Score
              </div>
            </div>

            <div style={{ 
              display: 'flex', 
              alignItems: 'center', 
              gap: '0.75rem',
              padding: 'clamp(0.75rem, 2vw, 1rem)',
              background: 'rgba(255,255,255,0.05)',
              borderRadius: '10px',
              borderLeft: `5px solid ${risk.color}`,
              boxShadow: `0 0 15px ${risk.color}1a`
            }}>
              <div style={{ 
                fontSize: 'clamp(1.2rem, 3vw, 1.5rem)', 
                fontWeight: '700',
                color: risk.color 
              }}>
                {risk.level}
              </div>
              <div style={{ fontSize: 'clamp(0.9rem, 2.5vw, 1rem)', color: '#ffffff' }}>
                Risk Level
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// Weather Data Display Component
const WeatherData = ({ weatherData }) => {
  if (!weatherData) return null;

  return (
    <div style={{
      background: '#000000',
      borderRadius: '12px',
      border: '1px solid #ffffff',
      padding: '1.5rem',
      marginBottom: '1.5rem'
    }}>
      <h4 style={{
        fontSize: 'clamp(1.2rem, 3vw, 1.5rem)',
        fontWeight: '600',
        color: '#ffffff',
        marginBottom: '1rem',
        textAlign: 'center'
      }}>
        🌤️ Weather Conditions
      </h4>
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
        gap: '1rem'
      }}>
        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '8px',
          padding: '1rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🌡️</div>
          <div style={{ fontSize: '0.9rem', color: '#cccccc', marginBottom: '0.25rem' }}>Temperature</div>
          <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#ffffff' }}>
            {weatherData.avgtemp_c?.toFixed(1) || 'N/A'}°C
          </div>
        </div>
        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '8px',
          padding: '1rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>💧</div>
          <div style={{ fontSize: '0.9rem', color: '#cccccc', marginBottom: '0.25rem' }}>Precipitation</div>
          <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#ffffff' }}>
            {weatherData.total_precip_mm?.toFixed(1) || 'N/A'}mm
          </div>
        </div>
        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '8px',
          padding: '1rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>💨</div>
          <div style={{ fontSize: '0.9rem', color: '#cccccc', marginBottom: '0.25rem' }}>Humidity</div>
          <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#ffffff' }}>
            {weatherData.avg_humidity?.toFixed(1) || 'N/A'}%
          </div>
        </div>
        <div style={{
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '8px',
          padding: '1rem',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>🌪️</div>
          <div style={{ fontSize: '0.9rem', color: '#cccccc', marginBottom: '0.25rem' }}>Wind Speed</div>
          <div style={{ fontSize: '1.2rem', fontWeight: '600', color: '#ffffff' }}>
            {weatherData.wind_kph?.toFixed(1) || 'N/A'} km/h
          </div>
        </div>
      </div>
    </div>
  );
};

// Main Fire Prediction Component
const FirePrediction = () => {
  const [formData, setFormData] = useState({
    latitude: '',
    longitude: '',
    date: new Date().toISOString().split('T')[0] // Today's date as default
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error and result when user starts typing
    if (error) setError(null);
    if (result) setResult(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.latitude || !formData.longitude || !formData.date) {
      setError('Please fill in all fields');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fireAPI.predictFireRisk({
        latitude: parseFloat(formData.latitude),
        longitude: parseFloat(formData.longitude),
        date: formData.date
      });
      setResult(response);
    } catch (err) {
      console.error('Fire prediction error:', err);
      setError(err.response?.data?.error || err.message || 'Failed to get fire risk prediction');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: '2rem 0' }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        padding: '0 1rem'
      }}>
        <h2 style={{
          fontSize: 'clamp(1.5rem, 4vw, 2rem)',
          fontWeight: '600',
          color: '#ffffff',
          marginBottom: '0.5rem',
          textAlign: 'center'
        }}>
          Wildfire Risk Prediction
        </h2>
        <p style={{
          fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
          color: '#888888',
          textAlign: 'center',
          marginBottom: '2rem'
        }}>
          Enter location coordinates and date to analyze wildfire risk
        </p>

        {/* Input Form */}
        <div style={{
          background: '#000000',
          borderRadius: '12px',
          border: '1px solid #ffffff',
          padding: 'clamp(1.5rem, 4vw, 2rem)',
          marginBottom: '1.5rem'
        }}>
          <form onSubmit={handleSubmit}>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '1rem',
              marginBottom: '1.5rem'
            }}>
              <div>
                <label style={{
                  display: 'block',
                  fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                  fontWeight: '500',
                  color: '#ffffff',
                  marginBottom: '0.5rem'
                }}>
                  Latitude:
                </label>
                <input
                  type="number"
                  name="latitude"
                  value={formData.latitude}
                  onChange={handleInputChange}
                  placeholder="e.g., 37.7749"
                  step="any"
                  min="-90"
                  max="90"
                  style={{
                    width: '100%',
                    padding: '0.75rem 1rem',
                    fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                    background: '#000000',
                    border: '1px solid #ffffff',
                    borderRadius: '8px',
                    color: '#ffffff',
                    outline: 'none',
                    transition: 'border-color 0.2s ease'
                  }}
                />
              </div>
              <div>
                <label style={{
                  display: 'block',
                  fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                  fontWeight: '500',
                  color: '#ffffff',
                  marginBottom: '0.5rem'
                }}>
                  Longitude:
                </label>
                <input
                  type="number"
                  name="longitude"
                  value={formData.longitude}
                  onChange={handleInputChange}
                  placeholder="e.g., -122.4194"
                  step="any"
                  min="-180"
                  max="180"
                  style={{
                    width: '100%',
                    padding: '0.75rem 1rem',
                    fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                    background: '#000000',
                    border: '1px solid #ffffff',
                    borderRadius: '8px',
                    color: '#ffffff',
                    outline: 'none',
                    transition: 'border-color 0.2s ease'
                  }}
                />
              </div>
              <div>
                <label style={{
                  display: 'block',
                  fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                  fontWeight: '500',
                  color: '#ffffff',
                  marginBottom: '0.5rem'
                }}>
                  Date:
                </label>
                <input
                  type="date"
                  name="date"
                  value={formData.date}
                  onChange={handleInputChange}
                  style={{
                    width: '100%',
                    padding: '0.75rem 1rem',
                    fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                    background: '#000000',
                    border: '1px solid #ffffff',
                    borderRadius: '8px',
                    color: '#ffffff',
                    outline: 'none',
                    transition: 'border-color 0.2s ease'
                  }}
                />
              </div>
            </div>

            <div style={{ textAlign: 'center' }}>
              <button 
                type="submit"
                disabled={loading || !formData.latitude || !formData.longitude || !formData.date}
                style={{
                  background: loading || !formData.latitude || !formData.longitude || !formData.date ? '#000000' : '#ffffff',
                  color: loading || !formData.latitude || !formData.longitude || !formData.date ? '#ffffff' : '#000000',
                  border: '1px solid #ffffff',
                  padding: '0.75rem 2rem',
                  fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                  fontWeight: '500',
                  borderRadius: '8px',
                  cursor: loading || !formData.latitude || !formData.longitude || !formData.date ? 'not-allowed' : 'pointer',
                  transition: 'all 0.2s ease',
                  opacity: loading || !formData.latitude || !formData.longitude || !formData.date ? 0.6 : 1,
                  minWidth: '140px'
                }}
              >
                {loading ? 'Analyzing...' : 'Predict Fire Risk'}
              </button>
            </div>
          </form>
        </div>

        {/* Error Display */}
        {error && (
          <div style={{
            background: '#000000',
            border: '1px solid #ff4757',
            borderRadius: '8px',
            padding: '1rem',
            marginBottom: '1.5rem',
            color: '#ff4757'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>⚠️</span>
              <strong>Error:</strong> {error}
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <>
            <FireRiskLevel riskScore={result.risk_score} />
            {result.weather_data && <WeatherData weatherData={result.weather_data} />}
          </>
        )}

        {/* Information Section */}
        <div style={{
          background: 'linear-gradient(135deg, #000000 0%, #111111 50%, #000000 100%)',
          borderRadius: '20px',
          border: '2px solid #ffffff',
          padding: 'clamp(2rem, 5vw, 3rem)',
          marginTop: '2rem',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(255,255,255,0.1)'
        }}>
          <div style={{
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: `
              linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
              linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px)
            `,
            backgroundSize: '50px 50px',
            animation: 'gridMove 20s linear infinite'
          }}></div>
          
          <div style={{ position: 'relative', zIndex: 2 }}>
            <h3 style={{
              fontSize: 'clamp(1.5rem, 4vw, 2rem)',
              fontWeight: '700',
              color: '#ffffff',
              marginBottom: '1rem',
              textAlign: 'center'
            }}>
              🔥 How Fire Risk Prediction Works
            </h3>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
              gap: '1.5rem',
              marginTop: '2rem'
            }}>
              <div style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '12px',
                padding: '1.5rem',
                border: '1px solid rgba(255,255,255,0.1)'
              }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem', textAlign: 'center' }}>🌡️</div>
                <h4 style={{
                  fontSize: '1.2rem',
                  fontWeight: '600',
                  color: '#ffffff',
                  marginBottom: '0.5rem',
                  textAlign: 'center'
                }}>
                  Weather Analysis
                </h4>
                <p style={{
                  fontSize: '0.9rem',
                  color: '#cccccc',
                  lineHeight: '1.5',
                  textAlign: 'center'
                }}>
                  Analyzes temperature, humidity, precipitation, wind speed, and atmospheric pressure
                </p>
              </div>
              <div style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '12px',
                padding: '1.5rem',
                border: '1px solid rgba(255,255,255,0.1)'
              }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem', textAlign: 'center' }}>🤖</div>
                <h4 style={{
                  fontSize: '1.2rem',
                  fontWeight: '600',
                  color: '#ffffff',
                  marginBottom: '0.5rem',
                  textAlign: 'center'
                }}>
                  AI Model
                </h4>
                <p style={{
                  fontSize: '0.9rem',
                  color: '#cccccc',
                  lineHeight: '1.5',
                  textAlign: 'center'
                }}>
                  Machine learning model trained on historical fire data and weather patterns
                </p>
              </div>
              <div style={{
                background: 'rgba(255,255,255,0.05)',
                borderRadius: '12px',
                padding: '1.5rem',
                border: '1px solid rgba(255,255,255,0.1)'
              }}>
                <div style={{ fontSize: '2rem', marginBottom: '1rem', textAlign: 'center' }}>📊</div>
                <h4 style={{
                  fontSize: '1.2rem',
                  fontWeight: '600',
                  color: '#ffffff',
                  marginBottom: '0.5rem',
                  textAlign: 'center'
                }}>
                  Risk Assessment
                </h4>
                <p style={{
                  fontSize: '0.9rem',
                  color: '#cccccc',
                  lineHeight: '1.5',
                  textAlign: 'center'
                }}>
                  Generates a probability score from 0-1 indicating wildfire risk likelihood
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default FirePrediction;
