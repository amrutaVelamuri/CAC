#!/usr/bin/env python3

import http.server
import socketserver
import json
import os

class AppHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(self.get_html().encode('utf-8'))
        else:
            super().do_GET()
    
    def get_html(self):
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Walking App</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta charset="UTF-8">
    <link rel="stylesheet" href="css/style.css">
</head>
<body>
    <div class="header">
        <h1>Smart Walking App</h1>
    </div>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('home')">Home</button>
        <button class="tab" onclick="showTab('exercise')">Exercise</button>
        <button class="tab" onclick="showTab('goals')">Goals</button>
        <button class="tab" onclick="showTab('progress')">Progress</button>
    </div>
    
    <div class="content">
        <div id="home" class="tab-content">
            <div class="section-title">Today's Overview</div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Steps Today</div>
                    <div class="metric-value" id="steps-value">0</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric-subtitle">Goal: 5,000 steps</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Exercise Minutes</div>
                    <div class="metric-value" id="exercise-value">0</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric-subtitle">Goal: 20 minutes</div>
            </div>
            
            <div class="streak-card">
                <div class="streak-number" id="streak-days">0</div>
                <div class="streak-label">Day Streak</div>
            </div>
            
            <div class="level-card">
                <div class="level-number" id="level">1</div>
                <div class="level-label">Level</div>
                <div class="xp-bar">
                    <div class="xp-fill" style="width: 0%"></div>
                </div>
                <div style="font-size: 12px; margin-top: 5px;">0/100 XP</div>
            </div>
            
            <button class="button" onclick="startWalking()">Start Walking Session</button>
            <button class="button" onclick="resetDay()" style="background: #dc3545;">Reset Day</button>
            
            <div class="notification">
                <strong>Tip:</strong> Start with small goals and build up your daily routine!
            </div>
        </div>
        
        <div id="exercise" class="tab-content hidden">
            <div class="section-title">Exercise Activities</div>
            
            <div class="exercise-item">
                <div class="exercise-name">Quick Walk</div>
                <div class="exercise-details">Duration: 10-15 min • Steps: 500-1000 • Easy</div>
                <button class="button" onclick="startExercise('Quick Walk')">Start Walk</button>
            </div>
            
            <div class="exercise-item">
                <div class="exercise-name">Balance Training</div>
                <div class="exercise-details">Duration: 5-10 min • Focus: Stability • Easy</div>
                <button class="button" onclick="startExercise('Balance Training')">Start Training</button>
            </div>
            
            <div class="exercise-item">
                <div class="exercise-name">Park Circuit</div>
                <div class="exercise-details">Duration: 20-30 min • Steps: 2000+ • Medium</div>
                <button class="button" onclick="startExercise('Park Circuit')">Start Circuit</button>
            </div>
            
            <div class="exercise-item">
                <div class="exercise-name">Strength Training</div>
                <div class="exercise-details">Duration: 15-25 min • Focus: Muscle • Medium</div>
                <button class="button" onclick="startExercise('Strength Training')">Start Training</button>
            </div>
        </div>
        
        <div id="goals" class="tab-content hidden">
            <div class="section-title">Achievements & Badges</div>
            
            <div class="badge-grid">
                <!-- Badges will be populated by JavaScript -->
            </div>
            
            <div class="section-title">Daily Goals</div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Steps Goal</div>
                    <div class="metric-value" id="steps-goal-display">0 / 5,000</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric-subtitle">Reward: 50 XP</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Exercise Goal</div>
                    <div class="metric-value" id="exercise-goal-display">0 / 20 min</div>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="metric-subtitle">Reward: 40 XP</div>
            </div>
            
            <div class="section-title">Weekly Challenges</div>
            
            <div class="exercise-item">
                <div class="exercise-name">Steps Challenge</div>
                <div class="exercise-details">Complete daily step goal 5 times this week</div>
                <div class="metric-subtitle">Progress: 0/5 days • Reward: 100 XP</div>
            </div>
            
            <div class="exercise-item">
                <div class="exercise-name">Exercise Challenge</div>
                <div class="exercise-details">Complete daily exercise goal 5 times this week</div>
                <div class="metric-subtitle">Progress: 0/5 days • Reward: 75 XP</div>
            </div>
        </div>
        
        <div id="progress" class="tab-content hidden">
            <div class="section-title">Weekly Progress</div>
            
            <div class="chart-container">
                <div class="chart-title">Steps This Week</div>
                <div class="chart-bar">
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                </div>
                <div class="day-labels">
                    <span>Mon</span>
                    <span>Tue</span>
                    <span>Wed</span>
                    <span>Thu</span>
                    <span>Fri</span>
                    <span>Sat</span>
                    <span>Sun</span>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">Exercise Minutes This Week</div>
                <div class="chart-bar">
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                    <div class="bar" style="height: 10%"></div>
                </div>
                <div class="day-labels">
                    <span>Mon</span>
                    <span>Tue</span>
                    <span>Wed</span>
                    <span>Thu</span>
                    <span>Fri</span>
                    <span>Sat</span>
                    <span>Sun</span>
                </div>
            </div>
            
            <div class="section-title">Stats Summary</div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Total Points Earned</div>
                    <div class="metric-value" id="total-points">0</div>
                </div>
                <div class="metric-subtitle">Keep going to earn more!</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-label">Current Level</div>
                    <div class="metric-value" id="current-level">1</div>
                </div>
                <div class="metric-subtitle">Level up by earning XP!</div>
            </div>
        </div>
    </div>
    
    <div class="achievement-popup" id="achievement-popup">
        <div class="achievement-icon">🏆</div>
        <div class="achievement-title">Achievement!</div>
        <div class="achievement-desc" id="achievement-desc">You've made progress!</div>
        <button class="button" onclick="closeAchievement()">Awesome!</button>
    </div>
    
    <script src="js/app.js"></script>
</body>
</html>
        """

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), AppHandler) as httpd:
        print(f"Mobile app running at http://localhost:{PORT}")
        print("Open in mobile browser or resize desktop browser to mobile size")
        httpd.serve_forever()
