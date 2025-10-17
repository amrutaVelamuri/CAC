import React, { useState, useEffect, useRef } from 'react';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const GaitTracker = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [sensorData, setSensorData] = useState({
    accelerometer: { x: 0, y: 0, z: 0 },
    gyroscope: { x: 0, y: 0, z: 0 },
    magnetometer: { x: 0, y: 0, z: 0 },
    orientation: { alpha: 0, beta: 0, gamma: 0 },
    heartRate: 0,
    stepCount: 0,
    cadence: 0,
    strideLength: 0,
    gps: { latitude: 0, longitude: 0, accuracy: 0 }
  });

  const [gaitMetrics, setGaitMetrics] = useState({
    cadence: 0,
    strideLength: 0,
    balanceScore: 0,
    gaitSpeed: 0,
    asymmetry: 0,
    variability: 0
  });

  const [isTracking, setIsTracking] = useState(false);
  const [trackingHistory, setTrackingHistory] = useState([]);
  const [userData, setUserData] = useState(null);
  const wsRef = useRef(null);
  const dataBuffer = useRef([]);

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket('ws://localhost:8765');
        
        wsRef.current.onopen = () => {
          console.log('Connected to WebSocket');
          setIsConnected(true);
        };

        wsRef.current.onmessage = (event) => {
          const data = JSON.parse(event.data);
          handleWebSocketMessage(data);
        };

        wsRef.current.onclose = () => {
          console.log('WebSocket disconnected');
          setIsConnected(false);
          // Attempt to reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        wsRef.current.onerror = (error) => {
          console.error('WebSocket error:', error);
          setIsConnected(false);
        };
      } catch (error) {
        console.error('Failed to connect to WebSocket:', error);
        setIsConnected(false);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  const handleWebSocketMessage = (data) => {
    switch (data.type) {
      case 'sensor_update':
        setSensorData(data.data);
        updateGaitMetrics(data.data);
        break;
      case 'user_data':
        setUserData(data.data);
        break;
      case 'user_data_update':
        setUserData(data.data);
        break;
      default:
        console.log('Unknown message type:', data.type);
    }
  };

  const updateGaitMetrics = (newData) => {
    // Add to buffer
    dataBuffer.current.push({
      ...newData,
      timestamp: Date.now()
    });

    // Keep only last 100 readings
    if (dataBuffer.current.length > 100) {
      dataBuffer.current = dataBuffer.current.slice(-100);
    }

    // Calculate gait metrics
    const metrics = calculateGaitMetrics(dataBuffer.current);
    setGaitMetrics(metrics);
  };

  const calculateGaitMetrics = (data) => {
    if (data.length < 10) return gaitMetrics;

    // Calculate cadence (steps per minute)
    const recentSteps = data.slice(-10).map(d => d.step_count);
    const stepRate = recentSteps[recentSteps.length - 1] - recentSteps[0];
    const cadence = Math.max(0, stepRate * 6); // Convert to steps per minute

    // Calculate stride length (simplified)
    const avgAcceleration = data.slice(-10).reduce((sum, d) => {
      const accel = Math.sqrt(d.accelerometer.x**2 + d.accelerometer.y**2 + d.accelerometer.z**2);
      return sum + accel;
    }, 0) / 10;
    
    const strideLength = Math.max(0, avgAcceleration * 0.5); // Simplified calculation

    // Calculate balance score
    const gyroVariance = data.slice(-10).reduce((sum, d) => {
      const gyro = Math.sqrt(d.gyroscope.x**2 + d.gyroscope.y**2 + d.gyroscope.z**2);
      return sum + gyro;
    }, 0) / 10;
    const balanceScore = Math.max(0, 100 - gyroVariance * 10);

    // Calculate gait speed
    const gaitSpeed = Math.max(0, cadence * strideLength / 100);

    // Calculate asymmetry (left vs right step differences)
    const leftSteps = data.filter((_, i) => i % 2 === 0).length;
    const rightSteps = data.filter((_, i) => i % 2 === 1).length;
    const asymmetry = Math.abs(leftSteps - rightSteps) / Math.max(leftSteps, rightSteps, 1) * 100;

    // Calculate variability
    const stepIntervals = [];
    for (let i = 1; i < data.length; i++) {
      stepIntervals.push(data[i].timestamp - data[i-1].timestamp);
    }
    const avgInterval = stepIntervals.reduce((sum, interval) => sum + interval, 0) / stepIntervals.length;
    const variance = stepIntervals.reduce((sum, interval) => sum + Math.pow(interval - avgInterval, 2), 0) / stepIntervals.length;
    const variability = Math.sqrt(variance) / avgInterval * 100;

    return {
      cadence: Math.round(cadence * 10) / 10,
      strideLength: Math.round(strideLength * 100) / 100,
      balanceScore: Math.round(balanceScore * 10) / 10,
      gaitSpeed: Math.round(gaitSpeed * 100) / 100,
      asymmetry: Math.round(asymmetry * 10) / 10,
      variability: Math.round(variability * 10) / 10
    };
  };

  const startTracking = () => {
    setIsTracking(true);
    setTrackingHistory([]);
    dataBuffer.current = [];
    
    // Request user data
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'get_user_data',
        user_id: 'default_user'
      }));
    }
  };

  const stopTracking = () => {
    setIsTracking(false);
    
    // Save tracking session
    const sessionData = {
      timestamp: new Date().toISOString(),
      duration: trackingHistory.length,
      metrics: gaitMetrics,
      sensorData: dataBuffer.current
    };
    
    setTrackingHistory(prev => [...prev, sessionData]);
    
    // Send data to server
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'update_user_data',
        user_id: 'default_user',
        data: {
          gait_metrics: gaitMetrics,
          last_tracking_session: sessionData
        }
      }));
    }
  };

  const addSteps = (steps) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'add_steps',
        user_id: 'default_user',
        steps: steps
      }));
    }
  };

  const addExercise = (minutes) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'add_exercise',
        user_id: 'default_user',
        minutes: minutes
      }));
    }
  };

  // Chart data
  const chartData = {
    labels: trackingHistory.map((_, i) => `Session ${i + 1}`),
    datasets: [
      {
        label: 'Cadence',
        data: trackingHistory.map(session => session.metrics.cadence),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1
      },
      {
        label: 'Balance Score',
        data: trackingHistory.map(session => session.metrics.balanceScore),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1
      }
    ]
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Gait Metrics Over Time'
      }
    },
    scales: {
      y: {
        beginAtZero: true
      }
    }
  };

  return (
    <div style={{ padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h1>Gait Tracker</h1>
      
      {/* Connection Status */}
      <div style={{ marginBottom: '20px' }}>
        <span style={{ 
          color: isConnected ? 'green' : 'red',
          fontWeight: 'bold'
        }}>
          {isConnected ? '🟢 Connected' : '🔴 Disconnected'}
        </span>
      </div>

      {/* User Data Display */}
      {userData && (
        <div style={{ 
          backgroundColor: '#f0f0f0', 
          padding: '15px', 
          borderRadius: '8px',
          marginBottom: '20px'
        }}>
          <h3>User Stats</h3>
          <p>Level: {userData.level} | XP: {userData.xp.toFixed(1)}</p>
          <p>Steps Today: {userData.steps_today} / {userData.steps_goal}</p>
          <p>Exercise: {userData.exercise_minutes} / {userData.exercise_goal} min</p>
          <p>Streak: {userData.streak_days} days</p>
        </div>
      )}

      {/* Control Buttons */}
      <div style={{ marginBottom: '20px' }}>
        <button 
          onClick={startTracking}
          disabled={!isConnected || isTracking}
          style={{
            backgroundColor: isTracking ? '#ccc' : '#4CAF50',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '4px',
            marginRight: '10px',
            cursor: isTracking ? 'not-allowed' : 'pointer'
          }}
        >
          {isTracking ? 'Tracking...' : 'Start Tracking'}
        </button>
        
        <button 
          onClick={stopTracking}
          disabled={!isTracking}
          style={{
            backgroundColor: !isTracking ? '#ccc' : '#f44336',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '4px',
            marginRight: '10px',
            cursor: !isTracking ? 'not-allowed' : 'pointer'
          }}
        >
          Stop Tracking
        </button>

        <button 
          onClick={() => addSteps(100)}
          disabled={!isConnected}
          style={{
            backgroundColor: !isConnected ? '#ccc' : '#2196F3',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '4px',
            marginRight: '10px',
            cursor: !isConnected ? 'not-allowed' : 'pointer'
          }}
        >
          Add 100 Steps
        </button>

        <button 
          onClick={() => addExercise(10)}
          disabled={!isConnected}
          style={{
            backgroundColor: !isConnected ? '#ccc' : '#FF9800',
            color: 'white',
            padding: '10px 20px',
            border: 'none',
            borderRadius: '4px',
            cursor: !isConnected ? 'not-allowed' : 'pointer'
          }}
        >
          Add 10 Min Exercise
        </button>
      </div>

      {/* Real-time Sensor Data */}
      <div style={{ 
        display: 'grid', 
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
        gap: '15px',
        marginBottom: '20px'
      }}>
        <div style={{ backgroundColor: '#e3f2fd', padding: '15px', borderRadius: '8px' }}>
          <h4>Accelerometer</h4>
          <p>X: {sensorData.accelerometer.x.toFixed(2)}</p>
          <p>Y: {sensorData.accelerometer.y.toFixed(2)}</p>
          <p>Z: {sensorData.accelerometer.z.toFixed(2)}</p>
        </div>
        
        <div style={{ backgroundColor: '#f3e5f5', padding: '15px', borderRadius: '8px' }}>
          <h4>Gyroscope</h4>
          <p>X: {sensorData.gyroscope.x.toFixed(2)}</p>
          <p>Y: {sensorData.gyroscope.y.toFixed(2)}</p>
          <p>Z: {sensorData.gyroscope.z.toFixed(2)}</p>
        </div>
        
        <div style={{ backgroundColor: '#e8f5e8', padding: '15px', borderRadius: '8px' }}>
          <h4>Health Data</h4>
          <p>Heart Rate: {sensorData.heartRate} bpm</p>
          <p>Step Count: {sensorData.stepCount}</p>
          <p>Cadence: {sensorData.cadence.toFixed(1)}</p>
        </div>
      </div>

      {/* Gait Metrics */}
      <div style={{ 
        backgroundColor: '#fff3e0', 
        padding: '20px', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3>Current Gait Metrics</h3>
        <div style={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', 
          gap: '15px'
        }}>
          <div>
            <strong>Cadence:</strong> {gaitMetrics.cadence} steps/min
          </div>
          <div>
            <strong>Stride Length:</strong> {gaitMetrics.strideLength} m
          </div>
          <div>
            <strong>Balance Score:</strong> {gaitMetrics.balanceScore}/100
          </div>
          <div>
            <strong>Gait Speed:</strong> {gaitMetrics.gaitSpeed} m/s
          </div>
          <div>
            <strong>Asymmetry:</strong> {gaitMetrics.asymmetry}%
          </div>
          <div>
            <strong>Variability:</strong> {gaitMetrics.variability}%
          </div>
        </div>
      </div>

      {/* Charts */}
      {trackingHistory.length > 0 && (
        <div style={{ marginBottom: '20px' }}>
          <h3>Tracking History</h3>
          <div style={{ height: '300px' }}>
            <Line data={chartData} options={chartOptions} />
          </div>
        </div>
      )}

      {/* Session History */}
      {trackingHistory.length > 0 && (
        <div>
          <h3>Session History</h3>
          <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
            {trackingHistory.map((session, index) => (
              <div key={index} style={{ 
                backgroundColor: '#f5f5f5', 
                padding: '10px', 
                margin: '5px 0',
                borderRadius: '4px'
              }}>
                <strong>Session {index + 1}</strong> - {new Date(session.timestamp).toLocaleString()}
                <br />
                Duration: {session.duration} readings | 
                Cadence: {session.metrics.cadence} | 
                Balance: {session.metrics.balanceScore}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default GaitTracker;
