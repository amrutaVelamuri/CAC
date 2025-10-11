import React, { useState, useEffect, useRef } from 'react';
import { Activity, Heart, Droplets, Navigation, Zap, Watch, Moon, AlertCircle, Smartphone } from 'lucide-react';

export default function ComprehensiveGaitTracker() {
  const [tracking, setTracking] = useState(false);
  const [sensorData, setSensorData] = useState({
    accelerometer: { x: 0, y: 0, z: 0 },
    gyroscope: { alpha: 0, beta: 0, gamma: 0 },
    magnetometer: { x: 0, y: 0, z: 0 },
    orientation: { absolute: false, alpha: 0, beta: 0, gamma: 0 },
    heartRate: 0,
    stepCount: 0,
    cadence: 0,
    strideLength: 0,
    gps: { latitude: null, longitude: null, speed: null, accuracy: null }
  });
  
  const [availableSensors, setAvailableSensors] = useState({
    accelerometer: false,
    gyroscope: false,
    magnetometer: false,
    orientation: false,
    heartRate: false,
    gps: false
  });

  const [serverUrl, setServerUrl] = useState('ws://192.168.1.100:8765');
  const [connected, setConnected] = useState(false);
  const [permissionsGranted, setPermissionsGranted] = useState(false);

  const wsRef = useRef(null);
  const accelerometerRef = useRef(null);
  const gyroscopeRef = useRef(null);
  const magnetometerRef = useRef(null);
  const orientationRef = useRef(null);
  const gpsWatchRef = useRef(null);
  const stepDetectorRef = useRef({ lastTimestamp: 0, peaks: [], threshold: 1.2 });

  useEffect(() => {
    checkSensorAvailability();
    return cleanup;
  }, []);

  const checkSensorAvailability = () => {
    const sensors = {
      accelerometer: false,
      gyroscope: false,
      magnetometer: false,
      orientation: false,
      heartRate: false,
      gps: false
    };

    // Check Accelerometer
    if ('Accelerometer' in window) {
      sensors.accelerometer = true;
    } else if (window.DeviceMotionEvent) {
      sensors.accelerometer = true;
    }

    // Check Gyroscope
    if ('Gyroscope' in window) {
      sensors.gyroscope = true;
    } else if (window.DeviceOrientationEvent) {
      sensors.gyroscope = true;
    }

    // Check Magnetometer
    if ('Magnetometer' in window) {
      sensors.magnetometer = true;
    }

    // Check Orientation
    if (window.DeviceOrientationEvent) {
      sensors.orientation = true;
    }

    // Check GPS
    if ('geolocation' in navigator) {
      sensors.gps = true;
    }

    // Check Bluetooth (for heart rate)
    if (navigator.bluetooth) {
      sensors.heartRate = true;
    }

    setAvailableSensors(sensors);
  };

  const requestPermissions = async () => {
    try {
      // Request motion sensors permission (iOS 13+)
      if (typeof DeviceMotionEvent !== 'undefined' && typeof DeviceMotionEvent.requestPermission === 'function') {
        const motionPermission = await DeviceMotionEvent.requestPermission();
        if (motionPermission !== 'granted') {
          alert('Motion sensor permission denied');
          return false;
        }
      }

      // Request orientation permission (iOS 13+)
      if (typeof DeviceOrientationEvent !== 'undefined' && typeof DeviceOrientationEvent.requestPermission === 'function') {
        const orientationPermission = await DeviceOrientationEvent.requestPermission();
        if (orientationPermission !== 'granted') {
          alert('Orientation sensor permission denied');
          return false;
        }
      }

      setPermissionsGranted(true);
      return true;
    } catch (error) {
      console.error('Permission error:', error);
      alert('Failed to get sensor permissions. Some features may not work.');
      return false;
    }
  };

  const connectToServer = () => {
    try {
      const ws = new WebSocket(serverUrl);
      
      ws.onopen = () => {
        setConnected(true);
        ws.send(JSON.stringify({ 
          type: 'connect',
          device: 'gait_tracker',
          timestamp: Date.now()
        }));
      };
      
      ws.onclose = () => setConnected(false);
      ws.onerror = () => setConnected(false);
      
      wsRef.current = ws;
    } catch (error) {
      alert('Failed to connect to server. Check URL.');
    }
  };

  const startAccelerometer = () => {
    if ('Accelerometer' in window) {
      try {
        const accel = new window.Accelerometer({ frequency: 100 });
        accel.addEventListener('reading', () => {
          const data = {
            x: accel.x || 0,
            y: accel.y || 0,
            z: accel.z || 0,
            timestamp: Date.now()
          };
          
          setSensorData(prev => ({ ...prev, accelerometer: data }));
          detectStep(data);
          sendToServer('accelerometer', data);
        });
        accel.start();
        accelerometerRef.current = accel;
      } catch (error) {
        console.error('Accelerometer error:', error);
      }
    } else if (window.DeviceMotionEvent) {
      const handler = (event) => {
        const data = {
          x: event.accelerationIncludingGravity?.x || 0,
          y: event.accelerationIncludingGravity?.y || 0,
          z: event.accelerationIncludingGravity?.z || 0,
          timestamp: Date.now()
        };
        
        setSensorData(prev => ({ ...prev, accelerometer: data }));
        detectStep(data);
        sendToServer('accelerometer', data);
      };
      
      window.addEventListener('devicemotion', handler);
      accelerometerRef.current = handler;
    }
  };

  const startGyroscope = () => {
    if ('Gyroscope' in window) {
      try {
        const gyro = new window.Gyroscope({ frequency: 100 });
        gyro.addEventListener('reading', () => {
          const data = {
            alpha: gyro.x || 0,
            beta: gyro.y || 0,
            gamma: gyro.z || 0,
            timestamp: Date.now()
          };
          
          setSensorData(prev => ({ ...prev, gyroscope: data }));
          sendToServer('gyroscope', data);
        });
        gyro.start();
        gyroscopeRef.current = gyro;
      } catch (error) {
        console.error('Gyroscope error:', error);
      }
    } else if (window.DeviceOrientationEvent) {
      const handler = (event) => {
        const data = {
          alpha: event.alpha || 0,
          beta: event.beta || 0,
          gamma: event.gamma || 0,
          timestamp: Date.now()
        };
        
        setSensorData(prev => ({ ...prev, gyroscope: data }));
        sendToServer('gyroscope', data);
      };
      
      window.addEventListener('deviceorientation', handler);
      gyroscopeRef.current = handler;
    }
  };

  const startMagnetometer = () => {
    if ('Magnetometer' in window) {
      try {
        const mag = new window.Magnetometer({ frequency: 60 });
        mag.addEventListener('reading', () => {
          const data = {
            x: mag.x || 0,
            y: mag.y || 0,
            z: mag.z || 0,
            timestamp: Date.now()
          };
          
          setSensorData(prev => ({ ...prev, magnetometer: data }));
          sendToServer('magnetometer', data);
        });
        mag.start();
        magnetometerRef.current = mag;
      } catch (error) {
        console.error('Magnetometer error:', error);
      }
    }
  };

  const startOrientation = () => {
    if (window.DeviceOrientationEvent) {
      const handler = (event) => {
        const data = {
          absolute: event.absolute || false,
          alpha: event.alpha || 0,
          beta: event.beta || 0,
          gamma: event.gamma || 0,
          timestamp: Date.now()
        };
        
        setSensorData(prev => ({ ...prev, orientation: data }));
        sendToServer('orientation', data);
      };
      
      window.addEventListener('deviceorientation', handler);
      orientationRef.current = handler;
    }
  };

  const startGPS = () => {
    if ('geolocation' in navigator) {
      const options = {
        enableHighAccuracy: true,
        timeout: 5000,
        maximumAge: 0
      };

      const watchId = navigator.geolocation.watchPosition(
        (position) => {
          const data = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
            speed: position.coords.speed,
            accuracy: position.coords.accuracy,
            altitude: position.coords.altitude,
            heading: position.coords.heading,
            timestamp: Date.now()
          };
          
          setSensorData(prev => ({ ...prev, gps: data }));
          sendToServer('gps', data);
        },
        (error) => {
          console.error('GPS error:', error);
        },
        options
      );
      
      gpsWatchRef.current = watchId;
    }
  };

  const connectHeartRateMonitor = async () => {
    if (!navigator.bluetooth) {
      alert('Bluetooth not supported');
      return;
    }

    try {
      const device = await navigator.bluetooth.requestDevice({
        filters: [{ services: ['heart_rate'] }],
        optionalServices: ['heart_rate']
      });

      const server = await device.gatt.connect();
      const service = await server.getPrimaryService('heart_rate');
      const characteristic = await service.getCharacteristic('heart_rate_measurement');
      
      await characteristic.startNotifications();
      characteristic.addEventListener('characteristicvaluechanged', (event) => {
        const value = event.target.value;
        const heartRate = value.getUint8(1);
        
        setSensorData(prev => ({ ...prev, heartRate }));
        sendToServer('heart_rate', { heart_rate: heartRate, timestamp: Date.now() });
      });
      
      alert(`Connected to ${device.name}`);
    } catch (error) {
      console.error('Bluetooth error:', error);
      alert('Failed to connect to heart rate monitor');
    }
  };

  const detectStep = (accelData) => {
    const magnitude = Math.sqrt(
      accelData.x * accelData.x +
      accelData.y * accelData.y +
      accelData.z * accelData.z
    );

    const detector = stepDetectorRef.current;
    const now = Date.now();

    if (magnitude > detector.threshold && (now - detector.lastTimestamp) > 300) {
      detector.peaks.push({ time: now, magnitude });
      detector.lastTimestamp = now;

      setSensorData(prev => {
        const newStepCount = prev.stepCount + 1;
        
        // Calculate cadence (steps per minute)
        const recentPeaks = detector.peaks.filter(p => now - p.time < 60000);
        const cadence = recentPeaks.length;
        
        // Estimate stride length (rough approximation)
        const strideLength = magnitude * 0.4; // Very rough estimate
        
        return {
          ...prev,
          stepCount: newStepCount,
          cadence: cadence,
          strideLength: strideLength
        };
      });
    }

    // Clean old peaks
    detector.peaks = detector.peaks.filter(p => now - p.time < 60000);
  };

  const sendToServer = (sensorType, data) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'sensor_data',
        sensor: sensorType,
        data: data,
        timestamp: Date.now()
      }));
    }
  };

  const startTracking = async () => {
    if (!connected) {
      alert('Please connect to server first');
      return;
    }

    const hasPermission = await requestPermissions();
    if (!hasPermission && !permissionsGranted) {
      return;
    }

    setTracking(true);

    if (availableSensors.accelerometer) startAccelerometer();
    if (availableSensors.gyroscope) startGyroscope();
    if (availableSensors.magnetometer) startMagnetometer();
    if (availableSensors.orientation) startOrientation();
    if (availableSensors.gps) startGPS();

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'tracking_start',
        sensors: availableSensors,
        timestamp: Date.now()
      }));
    }
  };

  const stopTracking = () => {
    setTracking(false);

    if (accelerometerRef.current) {
      if (accelerometerRef.current.stop) {
        accelerometerRef.current.stop();
      } else {
        window.removeEventListener('devicemotion', accelerometerRef.current);
      }
    }

    if (gyroscopeRef.current) {
      if (gyroscopeRef.current.stop) {
        gyroscopeRef.current.stop();
      } else {
        window.removeEventListener('deviceorientation', gyroscopeRef.current);
      }
    }

    if (magnetometerRef.current && magnetometerRef.current.stop) {
      magnetometerRef.current.stop();
    }

    if (orientationRef.current) {
      window.removeEventListener('deviceorientation', orientationRef.current);
    }

    if (gpsWatchRef.current) {
      navigator.geolocation.clearWatch(gpsWatchRef.current);
    }

    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'tracking_stop',
        timestamp: Date.now()
      }));
    }
  };

  const cleanup = () => {
    if (tracking) {
      stopTracking();
    }
    if (wsRef.current) {
      wsRef.current.close();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white p-4">
      <div className="max-w-2xl mx-auto">
        <div className="text-center mb-6 pt-6">
          <Activity className="w-14 h-14 mx-auto mb-3 text-purple-400" />
          <h1 className="text-3xl font-bold mb-2">Comprehensive Gait Tracker</h1>
          <p className="text-slate-300 text-sm">Multi-sensor gait analysis system</p>
        </div>

        {/* Server Connection */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 mb-4 border border-slate-700">
          <label className="block text-sm text-slate-300 mb-2 font-semibold">WebSocket Server</label>
          <input
            type="text"
            value={serverUrl}
            onChange={(e) => setServerUrl(e.target.value)}
            disabled={connected}
            className="w-full bg-slate-700 rounded-lg px-4 py-2 text-white disabled:opacity-50 mb-3 border border-slate-600"
            placeholder="ws://192.168.1.100:8765"
          />
          <button
            onClick={connectToServer}
            disabled={connected}
            className={`w-full py-3 rounded-lg font-semibold transition-all ${
              connected 
                ? 'bg-green-600 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {connected ? '✓ Connected' : 'Connect to Server'}
          </button>
        </div>

        {/* Available Sensors */}
        <div className="bg-slate-800/50 backdrop-blur rounded-xl p-4 mb-4 border border-slate-700">
          <h3 className="text-sm font-bold mb-3 text-purple-300 flex items-center">
            <Smartphone className="w-4 h-4 mr-2" />
            Available Sensors
          </h3>
          <div className="grid grid-cols-2 gap-2">
            <SensorStatus icon={Activity} name="Accelerometer" available={availableSensors.accelerometer} />
            <SensorStatus icon={Zap} name="Gyroscope" available={availableSensors.gyroscope} />
            <SensorStatus icon={Navigation} name="Magnetometer" available={availableSensors.magnetometer} />
            <SensorStatus icon={Navigation} name="Orientation" available={availableSensors.orientation} />
            <SensorStatus icon={Heart} name="Heart Rate" available={availableSensors.heartRate} />
            <SensorStatus icon={Navigation} name="GPS" available={availableSensors.gps} />
          </div>

          {availableSensors.heartRate && (
            <button
              onClick={connectHeartRateMonitor}
              disabled={tracking}
              className="w-full mt-3 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:opacity-50 rounded-lg py-2 font-semibold text-sm"
            >
              Connect Heart Rate Monitor
            </button>
          )}
        </div>

        {/* Tracking Controls */}
        {!tracking ? (
          <button
            onClick={startTracking}
            disabled={!connected}
            className="w-full bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 disabled:from-slate-700 disabled:to-slate-600 disabled:opacity-50 rounded-xl p-5 font-bold text-lg flex items-center justify-center shadow-lg"
          >
            <Activity className="w-6 h-6 mr-2" />
            Start Gait Tracking
          </button>
        ) : (
          <button
            onClick={stopTracking}
            className="w-full bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 rounded-xl p-5 font-bold text-lg flex items-center justify-center shadow-lg"
          >
            <AlertCircle className="w-6 h-6 mr-2" />
            Stop Tracking
          </button>
        )}

        {/* Live Data Display */}
        {tracking && (
          <div className="mt-6 space-y-4">
            <div className="bg-gradient-to-br from-purple-900/40 to-blue-900/40 backdrop-blur rounded-xl p-5 border border-purple-500/30">
              <h3 className="text-lg font-bold mb-4 text-center text-purple-300">Live Sensor Data</h3>
              
              {/* Step Metrics */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <MetricCard label="Steps" value={sensorData.stepCount} />
                <MetricCard label="Cadence" value={`${sensorData.cadence}/min`} />
                <MetricCard label="Stride" value={`${sensorData.strideLength.toFixed(2)}m`} />
              </div>

              {/* Accelerometer */}
              <DataDisplay
                icon={Activity}
                label="Accelerometer"
                data={[
                  { label: 'X', value: sensorData.accelerometer.x.toFixed(2) },
                  { label: 'Y', value: sensorData.accelerometer.y.toFixed(2) },
                  { label: 'Z', value: sensorData.accelerometer.z.toFixed(2) }
                ]}
                color="purple"
              />

              {/* Gyroscope */}
              <DataDisplay
                icon={Zap}
                label="Gyroscope"
                data={[
                  { label: 'α', value: sensorData.gyroscope.alpha.toFixed(1) },
                  { label: 'β', value: sensorData.gyroscope.beta.toFixed(1) },
                  { label: 'γ', value: sensorData.gyroscope.gamma.toFixed(1) }
                ]}
                color="blue"
              />

              {/* Magnetometer */}
              {availableSensors.magnetometer && (
                <DataDisplay
                  icon={Navigation}
                  label="Magnetometer"
                  data={[
                    { label: 'X', value: sensorData.magnetometer.x.toFixed(1) },
                    { label: 'Y', value: sensorData.magnetometer.y.toFixed(1) },
                    { label: 'Z', value: sensorData.magnetometer.z.toFixed(1) }
                  ]}
                  color="green"
                />
              )}

              {/* GPS */}
              {sensorData.gps.latitude && (
                <div className="bg-slate-800/50 rounded-lg p-3 mt-3">
                  <div className="flex items-center mb-2">
                    <Navigation className="w-4 h-4 mr-2 text-green-400" />
                    <span className="text-sm font-semibold">GPS</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div>Lat: {sensorData.gps.latitude.toFixed(6)}</div>
                    <div>Lon: {sensorData.gps.longitude.toFixed(6)}</div>
                    <div>Speed: {sensorData.gps.speed ? `${sensorData.gps.speed.toFixed(1)} m/s` : 'N/A'}</div>
                    <div>Acc: {sensorData.gps.accuracy ? `${sensorData.gps.accuracy.toFixed(0)}m` : 'N/A'}</div>
                  </div>
                </div>
              )}

              {/* Heart Rate */}
              {sensorData.heartRate > 0 && (
                <div className="bg-slate-800/50 rounded-lg p-3 mt-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center">
                      <Heart className="w-4 h-4 mr-2 text-red-400" />
                      <span className="text-sm font-semibold">Heart Rate</span>
                    </div>
                    <span className="text-2xl font-bold text-red-400">{sensorData.heartRate} BPM</span>
                  </div>
                </div>
              )}
            </div>

            <div className="bg-blue-900/20 border border-blue-500/50 rounded-lg p-3 text-xs text-center">
              <p className="mb-1">📱 Keep phone in pocket or mounted on leg</p>
              <p>🚶 Walk normally on flat surface for best results</p>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="mt-6 bg-slate-800/30 rounded-lg p-4 text-sm text-slate-400">
          <h4 className="font-semibold text-white mb-2">Setup Instructions:</h4>
          <ol className="space-y-1 list-decimal list-inside text-xs">
            <li>Start the Python WebSocket server first</li>
            <li>Enter server IP address (check Python console)</li>
            <li>Connect to server</li>
            <li>Optional: Connect Bluetooth heart rate monitor</li>
            <li>Grant sensor permissions when prompted</li>
            <li>Start tracking and begin walking</li>
            <li>For best results: mount phone on thigh or keep in pocket</li>
          </ol>
        </div>

        {/* Technical Info */}
        <div className="mt-4 bg-purple-900/20 border border-purple-500/50 rounded-lg p-4 text-xs">
          <h4 className="font-semibold text-purple-300 mb-2">📊 Sensor Data Collected</h4>
          <ul className="space-y-1 text-slate-300">
            <li><strong>Accelerometer:</strong> 3-axis acceleration (gait pattern, step detection)</li>
            <li><strong>Gyroscope:</strong> Angular velocity (rotation, balance, turning)</li>
            <li><strong>Magnetometer:</strong> Magnetic field (heading, direction changes)</li>
            <li><strong>GPS:</strong> Location, speed, stride length validation</li>
            <li><strong>Heart Rate:</strong> Cardiovascular response during walking</li>
            <li><strong>Orientation:</strong> Device angle relative to ground</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

function SensorStatus({ icon: Icon, name, available }) {
  return (
    <div className={`flex items-center p-2 rounded-lg ${available ? 'bg-green-900/30 border border-green-500/50' : 'bg-slate-700/30 border border-slate-600'}`}>
      <Icon className={`w-4 h-4 mr-2 ${available ? 'text-green-400' : 'text-slate-500'}`} />
      <div className="flex-1">
        <div className="text-xs font-semibold">{name}</div>
        <div className={`text-xs ${available ? 'text-green-400' : 'text-slate-500'}`}>
          {available ? '✓ Ready' : '✗ N/A'}
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value }) {
  return (
    <div className="bg-slate-800/60 rounded-lg p-3 text-center">
      <div className="text-xs text-slate-400 mb-1">{label}</div>
      <div className="text-xl font-bold text-purple-300">{value}</div>
    </div>
  );
}

function DataDisplay({ icon: Icon, label, data, color }) {
  const colorMap = {
    purple: 'text-purple-400',
    blue: 'text-blue-400',
    green: 'text-green-400',
    red: 'text-red-400'
  };

  return (
    <div className="bg-slate-800/50 rounded-lg p-3 mt-3">
      <div className="flex items-center mb-2">
        <Icon className={`w-4 h-4 mr-2 ${colorMap[color]}`} />
        <span className="text-sm font-semibold">{label}</span>
      </div>
      <div className="grid grid-cols-3 gap-3">
        {data.map((item, idx) => (
          <div key={idx} className="text-center">
            <div className="text-xs text-slate-400">{item.label}</div>
            <div className={`text-lg font-mono font-bold ${colorMap[color]}`}>{item.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}