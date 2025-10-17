#!/usr/bin/env python3
# Demo script for CAC Walking Tracker

import time
import json
import random
from datetime import datetime

def demo_sensor_data():
    """Generate demo sensor data"""
    return {
        "accelerometer": {
            "x": random.uniform(-2, 2),
            "y": random.uniform(-2, 2), 
            "z": random.uniform(8, 12)
        },
        "gyroscope": {
            "x": random.uniform(-0.5, 0.5),
            "y": random.uniform(-0.5, 0.5),
            "z": random.uniform(-0.5, 0.5)
        },
        "magnetometer": {
            "x": random.uniform(-50, 50),
            "y": random.uniform(-50, 50),
            "z": random.uniform(-50, 50)
        },
        "orientation": {
            "alpha": random.uniform(0, 360),
            "beta": random.uniform(-180, 180),
            "gamma": random.uniform(-90, 90)
        },
        "heart_rate": random.randint(60, 120),
        "step_count": random.randint(0, 10000),
        "cadence": random.uniform(100, 140),
        "stride_length": random.uniform(0.6, 0.8),
        "gps": {
            "latitude": random.uniform(37.7, 37.8),
            "longitude": random.uniform(-122.5, -122.4),
            "accuracy": random.uniform(1, 10)
        }
    }

def demo_gait_metrics():
    """Generate demo gait metrics"""
    return {
        "cadence": random.uniform(100, 140),
        "stride_length": random.uniform(0.6, 0.8),
        "balance_score": random.uniform(70, 95),
        "gait_speed": random.uniform(1.0, 1.5),
        "asymmetry": random.uniform(0, 15),
        "variability": random.uniform(5, 25)
    }

def main():
    """Run demo"""
    print("🚶 CAC Walking Tracker Demo")
    print("=" * 30)
    
    print("Generating demo data...")
    
    for i in range(5):
        print(f"\n--- Demo Reading {i+1} ---")
        
        sensor_data = demo_sensor_data()
        gait_metrics = demo_gait_metrics()
        
        print(f"Timestamp: {datetime.now().isoformat()}")
        print(f"Steps: {sensor_data['step_count']}")
        print(f"Heart Rate: {sensor_data['heart_rate']} bpm")
        print(f"Cadence: {gait_metrics['cadence']:.1f} steps/min")
        print(f"Balance Score: {gait_metrics['balance_score']:.1f}/100")
        
        time.sleep(1)
    
    print("\n✅ Demo completed!")
    print("\nTo run the full application:")
    print("  python3 start_cac.py")

if __name__ == "__main__":
    main()
