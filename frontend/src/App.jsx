import React, { useState } from 'react';
import FirePrediction from './components/FirePrediction';

// --- Global Styles (Including Keyframes for animations) ---
// In a real project, this would be in a global CSS file or defined using a library like styled-components
const globalStyles = `
  @keyframes pulse {
    0% {
      transform: scale(1);
      opacity: 1;
    }
    50% {
      transform: scale(1.05);
      opacity: 0.8;
    }
    100% {
      transform: scale(1);
      opacity: 1;
    }
  }

  @keyframes gridMove {
    from {
      background-position: 0 0;
    }
    to {
      background-position: 50px 50px;
    }
  }

  @keyframes orbit {
    from {
      transform: rotate(0deg) translate(80px) rotate(0deg);
    }
    to {
      transform: rotate(360deg) translate(80px) rotate(-360deg);
    }
  }

  @keyframes float {
    0% {
      transform: translateY(0px);
    }
    50% {
      transform: translateY(-10px);
    }
    100% {
      transform: translateY(0px);
    }
  }

  @keyframes glow {
    0%, 100% {
      opacity: 1;
    }
    50% {
      opacity: 0.7;
    }
  }
`;

// Inject keyframes into the document head
if (typeof document !== 'undefined') {
  const style = document.createElement('style');
  style.innerHTML = globalStyles;
  document.head.appendChild(style);
}

// --- Mock API Service (Replacing ./services/api) ---
// This mocks the backend service call for demonstration.
const satelliteAPI = {
  getSatelliteTile: (lat, lon) => {
    return new Promise((resolve, reject) => {
      // Simulate network delay
      setTimeout(() => {
        // Simple logic: lower latitude (closer to the equator) = higher risk
        const riskScore = Math.abs(lat / 90); // Normalize based on latitude
        let deforested = 0.4 + riskScore * 0.5; // Base risk + latitude factor

        // Ensure probabilities sum to 1 and are within [0, 1]
        deforested = Math.min(1, Math.max(0, deforested));
        const nonDeforested = 1 - deforested;

        // Add variance to make it less predictable
        const variance = (Math.random() * 0.1) - 0.05;
        deforested = Math.min(1, Math.max(0, deforested + variance));
        const nonDeforestedFinal = 1 - deforested;


        if (Math.random() < 0.9) { // 90% chance of success
          resolve({
            status: 200,
            data: {
              probabilities: {
                deforested: deforested.toFixed(2),
                non_deforested: nonDeforestedFinal.toFixed(2),
              },
            },
          });
        } else {
          // 10% chance of a simulated failure
          reject({
            response: {
              data: {
                error: 'Satellite link unstable. Please try another forest location.',
              },
            },
          });
        }
      }, 1500); // 1.5 second loading
    });
  },
};

// --- Navigation Component ---
const Navigation = ({ activeTab, setActiveTab }) => {
  const tabs = [
    { id: 'dashboard', label: 'Dashboard' },
    { id: 'about', label: 'About' },
    { id: 'contact', label: 'Contact' }
  ];

  return (
    <nav style={{
      background: '#000000',
      borderBottom: '1px solid #ffffff',
      padding: '0 1rem',
      position: 'sticky',
      top: 0,
      zIndex: 10
    }}>
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: '60px'
      }}>
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '0.5rem'
        }}>
          <div style={{
            width: '32px',
            height: '32px',
            background: '#4ade80',
            borderRadius: '6px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: '18px'
          }}>
            🌍
          </div>
          <h1 style={{
            fontSize: '1.5rem',
            fontWeight: '600',
            color: '#ffffff',
            margin: 0
          }}>
            Planetary
          </h1>
        </div>
        
        <div style={{
          display: 'flex',
          gap: '0.5rem'
        }}>
          {tabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '0.5rem 1rem',
                background: activeTab === tab.id ? '#ffffff' : 'transparent',
                border: activeTab === tab.id ? '1px solid #ffffff' : '1px solid transparent',
                borderRadius: '6px',
                color: activeTab === tab.id ? '#000000' : '#ffffff',
                cursor: 'pointer',
                fontSize: '0.9rem',
                fontWeight: '500',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                if (activeTab !== tab.id) {
                  e.target.style.background = '#ffffff';
                  e.target.style.color = '#000000';
                  e.target.style.border = '1px solid #ffffff';
                }
              }}
              onMouseLeave={(e) => {
                if (activeTab !== tab.id) {
                  e.target.style.background = 'transparent';
                  e.target.style.color = '#ffffff';
                  e.target.style.border = '1px solid transparent';
                }
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>
    </nav>
  );
};

// --- Circular Progress Component ---
const CircularProgress = ({ deforested, nonDeforested }) => {
  // Handle different data formats and ensure we have valid numbers
  const dVal = typeof deforested === 'number' ? deforested : parseFloat(deforested) || 0;
  const ndVal = typeof nonDeforested === 'number' ? nonDeforested : parseFloat(nonDeforested) || 0;
  
  // Re-normalize in case of API rounding errors, so total is 1
  const total = dVal + ndVal;
  const deforestedValue = total > 0 ? dVal / total : 0;
  const nonDeforestedValue = total > 0 ? ndVal / total : 0;

  const deforestedPercent = Math.round(deforestedValue * 100);
  const nonDeforestedPercent = Math.round(nonDeforestedValue * 100);
  
  // Determine which is majority for dynamic colors
  const isDeforestedMajority = deforestedValue > nonDeforestedValue;
  
  const radius = 40; // Adjusted radius for better fit in SVG
  const circumference = 2 * Math.PI * radius;

  // Primary: Majority (Higher percentage)
  const primaryValue = isDeforestedMajority ? deforestedValue : nonDeforestedValue;
  const primaryOffset = circumference - (primaryValue * circumference);
  const primaryColor = isDeforestedMajority ? '#ff4757' : '#4ade80';
  const primaryLabel = isDeforestedMajority ? 'Deforestation Risk' : 'Healthy Forest';
  const primaryPercent = Math.round(primaryValue * 100);

  // Secondary: Minority (Lower percentage)
  const secondaryValue = isDeforestedMajority ? nonDeforestedValue : deforestedValue;
  const secondaryOffset = primaryOffset - (secondaryValue * circumference); // Start after the primary arc
  const secondaryColor = isDeforestedMajority ? '#4ade80' : '#ff4757';
  const secondaryLabel = isDeforestedMajority ? 'Healthy Forest' : 'Deforestation Risk';
  const secondaryPercent = Math.round(secondaryValue * 100);

  // Final Risk Classification for the main display
  let riskClass = 'LOW';
  let riskColor = '#4ade80';
  if (deforestedPercent > 60) {
    riskClass = 'HIGH';
    riskColor = '#ff4757';
  } else if (deforestedPercent > 30) {
    riskClass = 'MODERATE';
    riskColor = '#facc15';
  }

  return (
    <div style={{ 
      display: 'flex', 
      alignItems: 'center', 
      gap: 'clamp(1rem, 4vw, 2rem)',
      justifyContent: 'center',
      flexWrap: 'wrap'
    }}>
      <div style={{ 
        position: 'relative', 
        width: 'clamp(120px, 25vw, 160px)', 
        height: 'clamp(120px, 25vw, 160px)',
        background: '#000000',
        borderRadius: '50%',
        padding: 'clamp(10px, 2vw, 15px)',
        border: '1px solid #ffffff'
      }}>
        <svg viewBox="0 0 100 100" width="100%" height="100%" style={{ transform: 'rotate(-90deg)' }}>
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="#ffffff"
            strokeWidth="8"
          />
          {/* Secondary arc (Minority) - Rendered first to sit underneath the primary arc */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={secondaryColor}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={secondaryOffset}
            strokeLinecap="butt"
            style={{ 
              transition: 'stroke-dashoffset 1s ease-out',
              filter: `drop-shadow(0 0 8px ${secondaryColor}40)`
            }}
          />
          {/* Primary arc (Majority) */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={primaryColor}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={primaryOffset}
            strokeLinecap="butt"
            style={{
              transition: 'stroke-dashoffset 1s ease-out',
              filter: `drop-shadow(0 0 10px ${primaryColor}40)`
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
            color: riskColor,
            marginBottom: '0.25rem',
            textShadow: `0 0 20px ${riskColor}60`
          }}>
            {deforestedPercent}%
          </div>
          <div style={{ 
            fontSize: 'clamp(0.7rem, 2vw, 0.8rem)', 
            color: '#ffffff',
            fontWeight: '500'
          }}>
            Risk: {riskClass}
          </div>
        </div>
      </div>
      
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        gap: 'clamp(0.75rem, 2vw, 1rem)',
        minWidth: 'clamp(200px, 40vw, 250px)'
      }}>
        {/* Primary Legend Item (Majority) */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.75rem',
          padding: 'clamp(0.75rem, 2vw, 1rem)',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '10px',
          borderLeft: `5px solid ${primaryColor}`,
          boxShadow: `0 0 15px ${primaryColor}1a`
        }}>
          <div style={{ 
            fontSize: 'clamp(1.2rem, 3vw, 1.5rem)', 
            fontWeight: '700',
            color: primaryColor 
          }}>
            {primaryPercent}%
          </div>
          <div style={{ fontSize: 'clamp(0.9rem, 2.5vw, 1rem)', color: '#ffffff' }}>
            {primaryLabel}
          </div>
        </div>

        {/* Secondary Legend Item (Minority) */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '0.75rem',
          padding: 'clamp(0.75rem, 2vw, 1rem)',
          background: 'rgba(255,255,255,0.05)',
          borderRadius: '10px',
          borderLeft: `5px solid ${secondaryColor}`,
          boxShadow: `0 0 15px ${secondaryColor}1a`
        }}>
          <div style={{ 
            fontSize: 'clamp(1.2rem, 3vw, 1.5rem)', 
            fontWeight: '700',
            color: secondaryColor 
          }}>
            {secondaryPercent}%
          </div>
          <div style={{ fontSize: 'clamp(0.9rem, 2.5vw, 1rem)', color: '#ffffff' }}>
            {secondaryLabel}
          </div>
        </div>
      </div>
    </div>
  );
};

// --- Dashboard Component ---
const Dashboard = () => {
  const [selectedForest, setSelectedForest] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const forests = [
    { name: 'Amazon Rainforest', lat: -3.4653, lon: -62.2159 },
    { name: 'Congo Rainforest', lat: 0.0, lon: 20.0 },
    { name: 'Borneo Rainforest', lat: 1.0, lon: 114.0 },
    { name: 'Southeast Asian Rainforest', lat: 12.0, lon: 105.0 },
    { name: 'Atlantic Forest', lat: -23.0, lon: -45.0 },
    { name: 'Daintree Rainforest', lat: -16.0, lon: 145.0 },
    { name: 'Tongass National Forest', lat: 58.0, lon: -135.0 },
    { name: 'Black Forest', lat: 48.0, lon: 8.0 }
  ];

  const handleSubmit = async () => {
    if (!selectedForest) {
      setError('Please select a forest');
      setResult(null);
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const forest = forests.find(f => f.name === selectedForest);
      // Ensure forest is found before calling API
      if (!forest) {
        setError('Invalid forest selection');
        return;
      }
      const response = await satelliteAPI.getSatelliteTile(forest.lat, forest.lon);
      setResult(response);
    } catch (err) {
      // Use optional chaining for safe error access
      setError(err.response?.data?.error || err.message || 'An unknown error occurred during analysis.');
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
          Forest Analysis Dashboard
        </h2>
        <p style={{
          fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
          color: '#888888',
          textAlign: 'center',
          marginBottom: '2rem'
        }}>
          Select a forest location to analyze deforestation risk
        </p>

        {/* Main Content Card */}
        <div style={{
          background: '#000000',
          borderRadius: '12px',
          border: '1px solid #ffffff',
          padding: 'clamp(1.5rem, 4vw, 2rem)',
          marginBottom: '1.5rem'
        }}>
          {/* Forest Selection */}
          <div style={{ marginBottom: '1.5rem' }}>
            <label style={{
              display: 'block',
              fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
              fontWeight: '500',
              color: '#ffffff',
              marginBottom: '0.75rem'
            }}>
              Select Forest Location:
            </label>
            <select 
              value={selectedForest} 
              onChange={(e) => {
                setSelectedForest(e.target.value);
                setError(null); // Clear error on new selection
                setResult(null); // Clear result on new selection
              }}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                background: '#000000',
                border: '1px solid #ffffff',
                borderRadius: '8px',
                color: '#ffffff',
                outline: 'none',
                transition: 'border-color 0.2s ease',
                cursor: 'pointer'
              }}
            >
              <option value="" style={{ background: '#000000', color: '#ffffff' }}>
                Choose a forest location...
              </option>
              {forests.map(forest => (
                <option key={forest.name} value={forest.name} style={{ background: '#000000', color: '#ffffff' }}>
                  {forest.name} ({forest.lat}, {forest.lon})
                </option>
              ))}
            </select>
          </div>

          {/* Action Button */}
          <div style={{ textAlign: 'center' }}>
            <button 
              onClick={handleSubmit} 
              disabled={loading || !selectedForest}
              style={{
                background: loading || !selectedForest ? '#000000' : '#ffffff',
                color: loading || !selectedForest ? '#ffffff' : '#000000',
                border: '1px solid #ffffff',
                padding: '0.75rem 2rem',
                fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                fontWeight: '500',
                borderRadius: '8px',
                cursor: loading || !selectedForest ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                opacity: loading || !selectedForest ? 0.6 : 1,
                minWidth: '140px'
              }}
            >
              {loading ? 'Processing...' : 'Analyze Forest'}
            </button>
          </div>
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
          <div style={{
            background: 'linear-gradient(135deg, #000000 0%, #111111 50%, #000000 100%)',
            borderRadius: '20px',
            border: '2px solid #4ade80', // Changed to green for a successful result highlight
            padding: 'clamp(2rem, 5vw, 3rem)',
            marginBottom: '2rem',
            position: 'relative',
            overflow: 'hidden',
            boxShadow: '0 0 30px rgba(74,222,128,0.3)'
          }}>
            {/* Animated Background */}
            <div style={{
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              background: `
                radial-gradient(circle at 20% 80%, rgba(74,222,128,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255,71,87,0.1) 0%, transparent 50%)
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
                textShadow: '0 0 30px rgba(255,255,255,0.5)',
                letterSpacing: '-0.02em'
              }}>
                📊 ANALYSIS RESULTS
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
                  🎯 Forest Health Assessment Complete
                </div>
                <div style={{
                  fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                  color: '#cccccc'
                }}>
                  AI-powered analysis of satellite imagery for {selectedForest}
                </div>
              </div>

              {result.data && result.data.probabilities && (
                <div style={{ marginBottom: '1.5rem' }}>
                  <h5 style={{
                    fontSize: 'clamp(1.3rem, 4vw, 1.8rem)',
                    fontWeight: '700',
                    color: '#ffffff',
                    marginBottom: '2rem',
                    textAlign: 'center',
                    textShadow: '0 0 20px rgba(255,255,255,0.3)'
                  }}>
                    🌍 Deforestation Risk Assessment
                  </h5>
                  <CircularProgress 
                    deforested={result.data.probabilities.deforested} 
                    nonDeforested={result.data.probabilities.non_deforested}
                  />
                </div>
              )}
            </div>
          </div>
        )}

        {/* Hero Section - How It Works */}
        <div style={{
          background: 'linear-gradient(135deg, #000000 0%, #111111 50%, #000000 100%)',
          borderRadius: '20px',
          border: '2px solid #ffffff',
          padding: 'clamp(3rem, 6vw, 4rem)',
          marginBottom: '2rem',
          position: 'relative',
          overflow: 'hidden',
          boxShadow: '0 20px 60px rgba(255,255,255,0.1)'
        }}>
          {/* Animated Background Grid */}
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
          
          {/* Floating Elements */}
          <div style={{
            position: 'absolute',
            top: '10%',
            right: '10%',
            width: '80px',
            height: '80px',
            background: 'radial-gradient(circle, rgba(74,222,128,0.2) 0%, transparent 70%)',
            borderRadius: '50%',
            animation: 'orbit 8s linear infinite'
          }}></div>
          <div style={{
            position: 'absolute',
            bottom: '15%',
            left: '15%',
            width: '60px',
            height: '60px',
            background: 'radial-gradient(circle, rgba(255,71,87,0.2) 0%, transparent 70%)',
            borderRadius: '50%',
            animation: 'orbit 6s linear infinite reverse'
          }}></div>
          <div style={{
            position: 'absolute',
            top: '50%',
            left: '5%',
            width: '40px',
            height: '40px',
            background: 'radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)',
            borderRadius: '50%',
            animation: 'float 4s ease-in-out infinite'
          }}></div>
          
          <div style={{ position: 'relative', zIndex: 2 }}>
            <h2 style={{
              fontSize: 'clamp(2.5rem, 6vw, 4rem)',
              fontWeight: '800',
              color: '#ffffff',
              marginBottom: '1rem',
              textAlign: 'center',
              textShadow: '0 0 30px rgba(255,255,255,0.5)',
              letterSpacing: '-0.02em'
            }}>
              🚀 HOW IT WORKS
            </h2>
            <p style={{
              fontSize: 'clamp(1.2rem, 3vw, 1.5rem)',
              color: '#ffffff',
              textAlign: 'center',
              marginBottom: '3rem',
              fontWeight: '300',
              textShadow: '0 0 20px rgba(255,255,255,0.3)',
              maxWidth: '600px',
              margin: '0 auto 3rem auto'
            }}>
              Cutting-Edge AI Technology Meets Environmental Conservation
            </p>
          
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '2rem',
              alignItems: 'center',
              marginTop: '2rem'
            }}>
              {/* Image Section */}
              <div style={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center'
              }}>
                <div style={{
                  width: '100%',
                  maxWidth: '400px',
                  height: '300px',
                  background: '#000000',
                  borderRadius: '12px',
                  border: '2px solid #ffffff',
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  position: 'relative',
                  overflow: 'hidden',
                  boxShadow: '0 0 30px rgba(255,255,255,0.1)'
                }}>
                  {/* Animated Satellite Icon */}
                  <div style={{
                    fontSize: '4rem',
                    marginBottom: '1rem',
                    animation: 'pulse 2s ease-in-out infinite',
                    filter: 'drop-shadow(0 0 10px rgba(255,255,255,0.3))'
                  }}>
                    🛰️
                  </div>
                  
                  {/* Animated Forest Visualization */}
                  <div style={{
                    display: 'flex',
                    gap: '0.5rem',
                    marginBottom: '1rem'
                  }}>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      background: '#4ade80',
                      borderRadius: '3px',
                      animation: 'glow 2s ease-in-out infinite',
                      boxShadow: '0 0 10px rgba(74, 222, 128, 0.5)'
                    }}></div>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      background: '#4ade80',
                      borderRadius: '3px',
                      animation: 'glow 2s ease-in-out infinite 0.2s',
                      boxShadow: '0 0 10px rgba(74, 222, 128, 0.5)'
                    }}></div>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      background: '#ff4757',
                      borderRadius: '3px',
                      animation: 'glow 2s ease-in-out infinite 0.4s',
                      boxShadow: '0 0 10px rgba(255, 71, 87, 0.5)'
                    }}></div>
                    <div style={{
                      width: '20px',
                      height: '20px',
                      background: '#4ade80',
                      borderRadius: '3px',
                      animation: 'glow 2s ease-in-out infinite 0.6s',
                      boxShadow: '0 0 10px rgba(74, 222, 128, 0.5)'
                    }}></div>
                  </div>
                  
                  <div style={{
                    fontSize: '0.9rem',
                    color: '#ffffff',
                    textAlign: 'center',
                    padding: '0 1rem',
                    fontWeight: '500',
                    textShadow: '0 0 10px rgba(255,255,255,0.3)'
                  }}>
                    Real-time Satellite Analysis
                  </div>
                </div>
              </div>

              {/* Process Steps */}
              <div style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '1.5rem'
              }}>
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem'
                }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    background: '#4ade80',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.9rem',
                    fontWeight: '600',
                    color: '#000000',
                    flexShrink: 0
                  }}>
                    1
                  </div>
                  <div>
                    <h3 style={{
                      fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
                      fontWeight: '600',
                      color: '#ffffff',
                      marginBottom: '0.5rem'
                    }}>
                      Satellite Data Collection
                    </h3>
                    <p style={{
                      fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                      color: '#cccccc',
                      lineHeight: '1.5'
                    }}>
                      High-resolution satellite imagery is captured for the selected forest location using advanced remote sensing technology.
                    </p>
                  </div>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem'
                }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    background: '#4ade80',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.9rem',
                    fontWeight: '600',
                    color: '#000000',
                    flexShrink: 0
                  }}>
                    2
                  </div>
                  <div>
                    <h3 style={{
                      fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
                      fontWeight: '600',
                      color: '#ffffff',
                      marginBottom: '0.5rem'
                    }}>
                      AI Analysis
                    </h3>
                    <p style={{
                      fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                      color: '#cccccc',
                      lineHeight: '1.5'
                    }}>
                      Machine learning algorithms analyze the imagery to detect patterns, changes, and potential deforestation indicators.
                    </p>
                  </div>
                </div>

                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '1rem'
                }}>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    background: '#4ade80',
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    fontSize: '0.9rem',
                    fontWeight: '600',
                    color: '#000000',
                    flexShrink: 0
                  }}>
                    3
                  </div>
                  <div>
                    <h3 style={{
                      fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
                      fontWeight: '600',
                      color: '#ffffff',
                      marginBottom: '0.5rem'
                    }}>
                      Risk Assessment
                    </h3>
                    <p style={{
                      fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
                      color: '#cccccc',
                      lineHeight: '1.5'
                    }}>
                      The system generates a probability score indicating the likelihood of deforestation, helping prioritize conservation efforts.
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Technology Stack */}
            <div style={{
              marginTop: '2rem',
              padding: '1.5rem',
              background: '#000000',
              borderRadius: '8px',
              border: '1px solid #ffffff'
            }}>
              <h3 style={{
                fontSize: 'clamp(1.2rem, 3vw, 1.5rem)',
                fontWeight: '600',
                color: '#ffffff',
                marginBottom: '1rem',
                textAlign: 'center'
              }}>
                Powered by Advanced Technology
              </h3>
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                gap: '1rem',
                textAlign: 'center'
              }}>
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🛰️</div>
                  <div style={{ fontSize: '0.9rem', color: '#cccccc' }}>Satellite Imagery</div>
                </div>
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🤖</div>
                  <div style={{ fontSize: '0.9rem', color: '#cccccc' }}>Machine Learning</div>
                </div>
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📊</div>
                  <div style={{ fontSize: '0.9rem', color: '#cccccc' }}>Data Analytics</div>
                </div>
                <div>
                  <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>🌍</div>
                  <div style={{ fontSize: '0.9rem', color: '#cccccc' }}>Environmental AI</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

// --- About Component ---
const About = () => (
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
        marginBottom: '1rem',
        textAlign: 'center'
      }}>
        About Planetary
      </h2>
      <div style={{
        background: '#000000',
        borderRadius: '12px',
        border: '1px solid #ffffff',
        padding: 'clamp(1.5rem, 4vw, 2rem)'
      }}>
        <p style={{
          fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
          color: '#cccccc',
          lineHeight: '1.6',
          marginBottom: '1rem'
        }}>
          Planetary is an advanced **AI-powered platform** for monitoring deforestation using satellite imagery. 
          Our system analyzes forest areas in real-time to detect and predict deforestation risks.
        </p>
        <p style={{
          fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
          color: '#cccccc',
          lineHeight: '1.6',
          marginBottom: '1rem'
        }}>
          Using cutting-edge machine learning algorithms, we process satellite data to provide accurate 
          assessments of forest health and deforestation probability.
        </p>
        <h3 style={{
          fontSize: 'clamp(1.2rem, 3vw, 1.5rem)',
          fontWeight: '600',
          color: '#ffffff',
          marginBottom: '0.75rem'
        }}>
          Features:
        </h3>
        <ul style={{
          fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
          color: '#cccccc',
          lineHeight: '1.6',
          paddingLeft: '1.5rem'
        }}>
          <li>Real-time satellite imagery analysis</li>
          <li>AI-powered deforestation detection</li>
          <li>Risk assessment and probability scoring</li>
          <li>Multiple forest location monitoring</li>
          <li>Interactive data visualization</li>
        </ul>
      </div>
    </div>
  </div>
);

// --- Contact Component ---
const Contact = () => (
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
        marginBottom: '1rem',
        textAlign: 'center'
      }}>
        Contact Us
      </h2>
      <div style={{
        background: '#000000',
        borderRadius: '12px',
        border: '1px solid #ffffff',
        padding: 'clamp(1.5rem, 4vw, 2rem)'
      }}>
        <div style={{
          display: 'grid',
          gap: '1.5rem',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))'
        }}>
          <div>
            <h3 style={{
              fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
              fontWeight: '600',
              color: '#ffffff',
              marginBottom: '0.5rem'
            }}>
              Email
            </h3>
            <p style={{
              fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
              color: '#cccccc'
            }}>
              contact@planetary.ai
            </p>
          </div>
          <div>
            <h3 style={{
              fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
              fontWeight: '600',
              color: '#ffffff',
              marginBottom: '0.5rem'
            }}>
              Phone
            </h3>
            <p style={{
              fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
              color: '#cccccc'
            }}>
              +1 (555) 123-4567
            </p>
          </div>
          <div>
            <h3 style={{
              fontSize: 'clamp(1rem, 2.5vw, 1.2rem)',
              fontWeight: '600',
              color: '#ffffff',
              marginBottom: '0.5rem'
            }}>
              Address
            </h3>
            <p style={{
              fontSize: 'clamp(0.9rem, 2.5vw, 1rem)',
              color: '#cccccc'
            }}>
              123 Forest Ave<br />
              Green City, GC 12345
            </p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// --- App Component (The missing piece) ---
// This is the main component that orchestrates the application.
export default function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  const renderContent = () => {
    switch (activeTab) {
      case 'dashboard':
        return <Dashboard />;
      case 'about':
        return <About />;
      case 'contact':
        return <Contact />;
      default:
        return <Dashboard />;
    }
  };

  // Add a global background style for the whole app
  return (
    <div style={{ 
      background: '#111111', 
      minHeight: '100vh', 
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Open Sans", "Helvetica Neue", sans-serif',
      color: '#ffffff'
    }}>
      <Navigation activeTab={activeTab} setActiveTab={setActiveTab} />
      {renderContent()}
      <FirePrediction/>
    </div>
  );
}