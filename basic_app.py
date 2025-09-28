#!/usr/bin/env python3

import http.server
import socketserver

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
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
            margin: 0; 
            padding: 0; 
            background: #f0f0f0; 
            font-size: 16px;
        }
        .header { 
            background: #007AFF; 
            color: white; 
            padding: 20px; 
            text-align: center; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }
        .tabs { 
            display: flex; 
            background: white;
            border-bottom: 1px solid #e0e0e0;
        }
        .tab { 
            flex: 1;
            padding: 15px; 
            background: white; 
            border: none; 
            cursor: pointer; 
            font-size: 16px;
            color: #666;
            border-bottom: 3px solid transparent;
        }
        .tab.active { 
            color: #007AFF; 
            border-bottom-color: #007AFF;
        }
        .content { 
            background: white; 
            padding: 20px; 
            min-height: calc(100vh - 140px);
        }
        .metric {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #007AFF;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 600;
            color: #333;
        }
        .button { 
            background: #007AFF; 
            color: white; 
            border: none; 
            padding: 15px 30px; 
            border-radius: 8px; 
            cursor: pointer; 
            font-size: 16px;
            width: 100%;
            margin: 10px 0;
        }
        .button:active {
            background: #0056CC;
        }
        .hidden { display: none; }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            margin: 20px 0 10px 0;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Smart Walking App</h1>
    </div>
    
    <div class="tabs">
        <button class="tab active" onclick="showTab('home')">Home</button>
        <button class="tab" onclick="showTab('analytics')">Analytics</button>
        <button class="tab" onclick="showTab('exercise')">Exercise</button>
    </div>
    
    <div class="content">
        <div id="home" class="tab-content">
            <div class="section-title">Dashboard</div>
            
            <div class="metric">
                <div class="metric-label">Health Score</div>
                <div class="metric-value">85/100</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Steps Today</div>
                <div class="metric-value">2,847</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Heart Rate</div>
                <div class="metric-value">72 bpm</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Sleep</div>
                <div class="metric-value">7h 24m</div>
            </div>
            
            <button class="button" onclick="alert('Route started!')">Start Walking Route</button>
        </div>
        
        <div id="analytics" class="tab-content hidden">
            <div class="section-title">Gait Analytics</div>
            
            <div class="metric">
                <div class="metric-label">Patient</div>
                <div class="metric-value">John Doe</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Stride Length</div>
                <div class="metric-value">68.2 cm</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Cadence</div>
                <div class="metric-value">119.5 spm</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Balance Score</div>
                <div class="metric-value">8.4/10</div>
            </div>
        </div>
        
        <div id="exercise" class="tab-content hidden">
            <div class="section-title">Exercise</div>
            
            <div class="metric">
                <div class="metric-label">Exercise Minutes</div>
                <div class="metric-value">12 min</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Walking Sessions</div>
                <div class="metric-value">2 sessions</div>
            </div>
            
            <div class="metric">
                <div class="metric-label">Calories Burned</div>
                <div class="metric-value">180 cal</div>
            </div>
            
            <button class="button" onclick="alert('Starting exercise...')">Start Exercise</button>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.add('hidden'));
            
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            document.getElementById(tabName).classList.remove('hidden');
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
        """

if __name__ == "__main__":
    PORT = 8000
    with socketserver.TCPServer(("", PORT), AppHandler) as httpd:
        print(f"Mobile app running at http://localhost:{PORT}")
        print("Open in mobile browser or resize desktop browser to mobile size")
        httpd.serve_forever()
