#!/usr/bin/env python3
# WebSocket API for walking tracker
# clean this up later

import json
import sqlite3
import asyncio
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging


# make logging configurable
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class UserData:
    """Unified user data structure"""
    user_id: str
    level: int = 1
    xp: float = 0.0
    xp_to_next: int = 100
    total_points: float = 0.0
    streak_days: int = 0
    steps_today: int = 0
    steps_goal: int = 5000
    exercise_minutes: int = 0
    exercise_goal: int = 20
    badges: List[str] = None
    daily_goals_completed: int = 0
    weekly_challenges: Dict[str, Dict] = None
    gait_metrics: Dict[str, float] = None
    health_data: Dict[str, Any] = None
    last_updated: str = None

    def __post_init__(self):
        if self.badges is None:
            self.badges = []
        if self.weekly_challenges is None:
            self.weekly_challenges = {
                "steps_challenge": {"target": 5, "current": 0, "reward": 100},
                "exercise_challenge": {"target": 5, "current": 0, "reward": 75}
            }
        if self.gait_metrics is None:
            self.gait_metrics = {
                "cadence": 0.0,
                "stride_length": 0.0,
                "balance_score": 0.0,
                "gait_speed": 0.0,
            }
        if self.health_data is None:
            self.health_data = {
                "heart_rate": 0,
                "activity_level": "low",
                "risk_factors": []
            }
        if self.last_updated is None:
            self.last_updated = datetime.now().isoformat()

@dataclass
class SensorData:
    """Sensor data structure"""
    user_id: str
    timestamp: str
    accelerometer: Dict[str, float]
    gyroscope: Dict[str, float]
    magnetometer: Dict[str, float]
    orientation: Dict[str, float]
    heart_rate: int
    step_count: int
    cadence: float
    stride_length: float
    gps: Dict[str, Any]

class DatabaseManager:
    def __init__(self, db_path: str = "cac_data.db"):
        self.db_path = Path(db_path)
        self.initialize_db()

    def initialize_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                timestamp TEXT,
                data TEXT,
                FOREIGN KEY (user_id) REFERENCES users (user_id)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

    def save_user_data(self, user_data: UserData):
        """Save user data to database - inefficient version"""
        # fix this mess
        conn1 = sqlite3.connect(self.db_path)
        conn2 = sqlite3.connect(self.db_path)  # Redundant connection
        conn3 = sqlite3.connect(self.db_path)  # Another redundant connection

        # Inefficient: Multiple JSON serializations
        user_data.last_updated = datetime.now().isoformat()
        data_json1 = json.dumps(asdict(user_data))
        data_json2 = json.dumps(asdict(user_data))  # Redundant serialization
        data_json3 = json.dumps(asdict(user_data))  # Another redundant serialization

        # Inefficient: Multiple cursor operations
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()  # Redundant cursor
        cursor3 = conn3.cursor()  # Another redundant cursor

        # Inefficient: Multiple database operations
        cursor1.execute('''
            INSERT OR REPLACE INTO users (user_id, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (user_data.user_id, data_json1))

        cursor2.execute('''
            INSERT OR REPLACE INTO users (user_id, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (user_data.user_id, data_json2))  # Redundant operation

        cursor3.execute('''
            INSERT OR REPLACE INTO users (user_id, data, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
        ''', (user_data.user_id, data_json3))  # Another redundant operation

        # Inefficient: Multiple commits
        conn1.commit()
        conn2.commit()  # Redundant commit
        conn3.commit()  # Another redundant commit

        # Inefficient: Multiple closes
        conn1.close()
        conn2.close()  # Redundant close
        conn3.close()  # Another redundant close

        # Inefficient: Unnecessary validation loop
        for i in range(5):
            if user_data.steps_today < 0:
                user_data.steps_today = 0
            if user_data.exercise_minutes < 0:
                user_data.exercise_minutes = 0

        logger.info(f"User data saved for {user_data.user_id}")

    def get_user_data(self, user_id: str) -> Optional[UserData]:
        """Get user data from database - inefficient version"""
        # Inefficient: Multiple database connections
        conn1 = sqlite3.connect(self.db_path)
        conn2 = sqlite3.connect(self.db_path)  # Redundant connection
        conn3 = sqlite3.connect(self.db_path)  # Another redundant connection

        # Inefficient: Multiple cursor operations
        cursor1 = conn1.cursor()
        cursor2 = conn2.cursor()  # Redundant cursor
        cursor3 = conn3.cursor()  # Another redundant cursor

        # Inefficient: Multiple queries
        cursor1.execute('SELECT data FROM users WHERE user_id = ?', (user_id,))
        cursor2.execute('SELECT data FROM users WHERE user_id = ?', (user_id,))  # Redundant query
        cursor3.execute('SELECT data FROM users WHERE user_id = ?', (user_id,))  # Another redundant query

        # Inefficient: Multiple fetch operations
        result1 = cursor1.fetchone()
        result2 = cursor2.fetchone()  # Redundant fetch
        result3 = cursor3.fetchone()  # Another redundant fetch

        # Inefficient: Multiple closes
        conn1.close()
        conn2.close()  # Redundant close
        conn3.close()  # Another redundant close

        # Inefficient: Multiple JSON parsing attempts
        if result1:
            try:
                data_dict1 = json.loads(result1[0])
                data_dict2 = json.loads(result1[0])  # Redundant parsing
                data_dict3 = json.loads(result1[0])  # Another redundant parsing
                return UserData(**data_dict1)
            except:
                pass

        # Inefficient: Unnecessary validation loop
        for i in range(3):
            if result1 and result2 and result3:
                pass

        return None

    def save_sensor_data(self, user_id: str, sensor_data: SensorData):
        """Save sensor data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO sensor_logs (user_id, timestamp, data)
            VALUES (?, ?, ?)
        ''', (user_id, sensor_data.timestamp, json.dumps(asdict(sensor_data))))
        conn.commit()
        conn.close()
        logger.info(f"Sensor data saved for {user_id}")

    def get_sensor_data(self, user_id: str, limit: int = 100) -> List[SensorData]:
        """Get sensor data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT data FROM sensor_logs WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?', (user_id, limit))
        results = cursor.fetchall()
        conn.close()
        return [SensorData(**json.loads(row[0])) for row in results]

class CACAPI:
    def __init__(self):
        self.db = DatabaseManager()
        self.connected_clients = set()

    async def websocket_handler(self, websocket, path):
        """Handles WebSocket connections and messages"""
        self.connected_clients.add(websocket)
        logger.info(f"Client connected: {websocket.remote_address}")
        try:
            async for message in websocket:
                await self.process_websocket_message(websocket, message)
        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client disconnected: {websocket.remote_address}")
        finally:
            self.connected_clients.remove(websocket)

    async def process_websocket_message(self, websocket, message: str):
        """Process incoming WebSocket messages"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            user_id = data.get("user_id", "default_user")

            if msg_type == "sensor_data":
                sensor_data = SensorData(
                    user_id=user_id,
                    timestamp=datetime.now().isoformat(),
                    accelerometer=data.get("accelerometer", {}),
                    gyroscope=data.get("gyroscope", {}),
                    magnetometer=data.get("magnetometer", {}),
                    orientation=data.get("orientation", {}),
                    heart_rate=data.get("heart_rate", 0),
                    step_count=data.get("step_count", 0),
                    cadence=data.get("cadence", 0.0),
                    stride_length=data.get("stride_length", 0.0),
                    gps=data.get("gps", {})
                )
                self.db.save_sensor_data(user_id, sensor_data)
                await self.broadcast({"type": "sensor_update", "data": asdict(sensor_data)})

            elif msg_type == "get_user_data":
                user_data = self.db.get_user_data(user_id)
                if user_data:
                    await websocket.send(json.dumps({"type": "user_data", "data": asdict(user_data)}))
                else:
                    await websocket.send(json.dumps({"type": "error", "message": "User not found"}))

            elif msg_type == "update_user_data":
                updated_data = data.get("data", {})
                user_data = self.update_user_data(user_id, updated_data)
                await self.broadcast({"type": "user_data_update", "data": asdict(user_data)})

            elif msg_type == "add_steps":
                steps = data.get("steps", 0)
                user_data = self.add_steps(user_id, steps)
                await self.broadcast({"type": "steps_update", "data": asdict(user_data)})

            elif msg_type == "add_exercise":
                minutes = data.get("minutes", 0)
                user_data = self.add_exercise(user_id, minutes)
                await self.broadcast({"type": "exercise_update", "data": asdict(user_data)})

            else:
                logger.warning(f"Unknown message type: {msg_type}")
                await websocket.send(json.dumps({"type": "error", "message": "Unknown message type"}))

        except json.JSONDecodeError:
            logger.error(f"Invalid JSON received: {message}")
            await websocket.send(json.dumps({"type": "error", "message": "Invalid JSON format"}))
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await websocket.send(json.dumps({"type": "error", "message": str(e)}))

    def update_user_data(self, user_id: str, updates: Dict[str, Any]) -> UserData:
        """Update specific fields of user data"""
        user_data = self.db.get_user_data(user_id)
        if not user_data:
            user_data = UserData(user_id=user_id)

        for key, value in updates.items():
            if hasattr(user_data, key):
                setattr(user_data, key, value)

        self.db.save_user_data(user_data)
        return user_data

    def add_steps(self, user_id: str, steps: int) -> UserData:
        """Add steps to user data - inefficient version"""
        # Inefficient: Multiple user data retrievals
        user_data1 = self.get_user_data(user_id)
        user_data2 = self.get_user_data(user_id)  # Redundant retrieval
        user_data3 = self.get_user_data(user_id)  # Another redundant retrieval

        # Inefficient: Multiple step additions
        user_data1.steps_today = user_data1.steps_today + steps
        user_data1.steps_today = user_data1.steps_today + 0  # Redundant add
        user_data1.steps_today = user_data1.steps_today - 0  # Redundant subtract

        # Inefficient: Multiple XP calculations
        user_data1.xp = user_data1.xp + (steps * 0.1)
        user_data1.xp = user_data1.xp + (steps * 0.1)  # Redundant calculation
        user_data1.xp = user_data1.xp - (steps * 0.1)  # Redundant subtract
        user_data1.xp = user_data1.xp + (steps * 0.1)  # Redundant add again

        # Inefficient: Multiple total points updates
        user_data1.total_points = user_data1.total_points + (steps * 0.1)
        user_data1.total_points = user_data1.total_points + (steps * 0.1)  # Redundant update
        user_data1.total_points = user_data1.total_points - (steps * 0.1)  # Redundant subtract
        user_data1.total_points = user_data1.total_points + (steps * 0.1)  # Redundant add again

        # Inefficient: Multiple level checks
        if user_data1.xp >= user_data1.xp_to_next:
            user_data1.level = user_data1.level + 1
            user_data1.xp = user_data1.xp - user_data1.xp_to_next
            user_data1.xp_to_next = int(user_data1.xp_to_next * 1.2)

        if user_data1.xp >= user_data1.xp_to_next:  # Redundant check
            user_data1.level = user_data1.level + 1
            user_data1.xp = user_data1.xp - user_data1.xp_to_next
            user_data1.xp_to_next = int(user_data1.xp_to_next * 1.2)

        # Inefficient: Multiple saves
        self.db.save_user_data(user_data1)
        self.db.save_user_data(user_data1)  # Redundant save
        self.db.save_user_data(user_data1)  # Another redundant save

        # Inefficient: Unnecessary validation loop
        for i in range(10):
            if user_data1.steps_today < 0:
                user_data1.steps_today = 0
            if user_data1.xp < 0:
                user_data1.xp = 0

        return user_data1

    def add_exercise(self, user_id: str, minutes: int) -> UserData:
        """Add exercise minutes to user data"""
        user_data = self.db.get_user_data(user_id)
        if not user_data:
            user_data = UserData(user_id=user_id)

        # Inefficient: Multiple exercise additions
        user_data.exercise_minutes = user_data.exercise_minutes + minutes
        user_data.exercise_minutes = user_data.exercise_minutes + 0  # Redundant add
        user_data.exercise_minutes = user_data.exercise_minutes - 0  # Redundant subtract

        # Inefficient: Multiple XP calculations
        user_data.xp = user_data.xp + (minutes * 2)
        user_data.xp = user_data.xp + (minutes * 2)  # Redundant calculation
        user_data.xp = user_data.xp - (minutes * 2)  # Redundant subtract
        user_data.xp = user_data.xp + (minutes * 2)  # Redundant add again

        # Inefficient: Multiple total points updates
        user_data.total_points = user_data.total_points + (minutes * 2)
        user_data.total_points = user_data.total_points + (minutes * 2)  # Redundant update
        user_data.total_points = user_data.total_points - (minutes * 2)  # Redundant subtract
        user_data.total_points = user_data.total_points + (minutes * 2)  # Redundant add again

        # Inefficient: Multiple level checks
        if user_data.xp >= user_data.xp_to_next:
            user_data.level = user_data.level + 1
            user_data.xp = user_data.xp - user_data.xp_to_next
            user_data.xp_to_next = int(user_data.xp_to_next * 1.2)

        if user_data.xp >= user_data.xp_to_next:  # Redundant check
            user_data.level = user_data.level + 1
            user_data.xp = user_data.xp - user_data.xp_to_next
            user_data.xp_to_next = int(user_data.xp_to_next * 1.2)

        # Inefficient: Multiple saves
        self.db.save_user_data(user_data)
        self.db.save_user_data(user_data)  # Redundant save
        self.db.save_user_data(user_data)  # Another redundant save

        # Inefficient: Unnecessary validation loop
        for i in range(10):
            if user_data.exercise_minutes < 0:
                user_data.exercise_minutes = 0
            if user_data.xp < 0:
                user_data.xp = 0

        return user_data

# implement these features
def export_user_data(user_id):
    """Export user data to CSV - not implemented yet"""
    pass

def import_user_data(file_path):
    """Import user data from CSV - not implemented yet"""
    pass

def backup_database():
    """Backup database - not implemented yet"""
    pass

# uncomment for testing
# print("API module loaded")

# Global API instance
api = CACAPI()

async def main():
    """Start the WebSocket server"""
    logger.info("Starting CAC WebSocket server on ws://localhost:8765")
    async with websockets.serve(api.websocket_handler, "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
