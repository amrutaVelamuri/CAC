class WalkingApp {
    constructor() {
        this.userData = this.loadUserData();
        this.badges = [
            { id: 'first_steps', name: 'First Steps', icon: '👟', requirement: 'Walk 100 steps', points: 10 },
            { id: 'step_master', name: 'Step Master', icon: '🚶‍♂️', requirement: 'Walk 1000 steps', points: 25 },
            { id: 'daily_walker', name: 'Daily Walker', icon: '🔥', requirement: 'Complete daily step goal', points: 50 },
            { id: 'week_warrior', name: 'Week Warrior', icon: '💪', requirement: 'Complete 7 daily goals', points: 100 },
            { id: 'exercise_starter', name: 'Exercise Starter', icon: '🏃‍♂️', requirement: 'Exercise for 10 minutes', points: 20 },
            { id: 'fitness_fanatic', name: 'Fitness Fanatic', icon: '🎯', requirement: 'Exercise for 60 minutes', points: 75 }
        ];
        this.init();
    }

    loadUserData() {
        const defaultData = {
            level: 1,
            xp: 0,
            xp_to_next: 100,
            total_points: 0,
            streak_days: 0,
            steps_today: 0,
            steps_goal: 5000,
            exercise_minutes: 0,
            exercise_goal: 20,
            badges: [],
            daily_goals_completed: 0,
            weekly_challenges: {
                steps_challenge: { target: 5, current: 0, reward: 100 },
                exercise_challenge: { target: 5, current: 0, reward: 75 }
            },
            history: {
                steps: [],
                exercise: [],
                dates: []
            }
        };

        const saved = localStorage.getItem('walkingAppData');
        return saved ? JSON.parse(saved) : defaultData;
    }

    saveUserData() {
        localStorage.setItem('walkingAppData', JSON.stringify(this.userData));
    }

    init() {
        this.updateUI();
        this.checkBadges();
        this.updateHistory();
    }

    addSteps(steps) {
        this.userData.steps_today += steps;
        this.addXP(steps * 0.1);
        this.checkDailyGoal();
        this.updateUI();
        this.saveUserData();
    }

    addExercise(minutes) {
        this.userData.exercise_minutes += minutes;
        this.addXP(minutes * 2);
        this.checkDailyGoal();
        this.updateUI();
        this.saveUserData();
    }

    addXP(amount) {
        this.userData.xp += amount;
        this.userData.total_points += amount;
        
        if (this.userData.xp >= this.userData.xp_to_next) {
            this.levelUp();
        }
    }

    levelUp() {
        this.userData.level++;
        this.userData.xp -= this.userData.xp_to_next;
        this.userData.xp_to_next = Math.floor(this.userData.xp_to_next * 1.2);
        this.showAchievement(`Level Up! You're now level ${this.userData.level}!`);
    }

    checkDailyGoal() {
        const stepsComplete = this.userData.steps_today >= this.userData.steps_goal;
        const exerciseComplete = this.userData.exercise_minutes >= this.userData.exercise_goal;
        
        if (stepsComplete && exerciseComplete && this.userData.daily_goals_completed === 0) {
            this.userData.daily_goals_completed = 1;
            this.userData.streak_days++;
            this.addXP(100);
            this.showAchievement('Daily goals completed! +100 XP');
        }
    }

    checkBadges() {
        this.badges.forEach(badge => {
            if (!this.userData.badges.includes(badge.id)) {
                let earned = false;
                
                switch(badge.id) {
                    case 'first_steps':
                        earned = this.userData.steps_today >= 100;
                        break;
                    case 'step_master':
                        earned = this.userData.steps_today >= 1000;
                        break;
                    case 'daily_walker':
                        earned = this.userData.steps_today >= this.userData.steps_goal;
                        break;
                    case 'week_warrior':
                        earned = this.userData.streak_days >= 7;
                        break;
                    case 'exercise_starter':
                        earned = this.userData.exercise_minutes >= 10;
                        break;
                    case 'fitness_fanatic':
                        earned = this.userData.exercise_minutes >= 60;
                        break;
                }
                
                if (earned) {
                    this.userData.badges.push(badge.id);
                    this.addXP(badge.points);
                    this.showAchievement(`Badge earned: ${badge.name}! +${badge.points} XP`);
                }
            }
        });
    }

    updateHistory() {
        const today = new Date().toISOString().split('T')[0];
        const lastEntry = this.userData.history.dates[this.userData.history.dates.length - 1];
        
        if (lastEntry !== today) {
            this.userData.history.dates.push(today);
            this.userData.history.steps.push(this.userData.steps_today);
            this.userData.history.exercise.push(this.userData.exercise_minutes);
            
            if (this.userData.history.dates.length > 7) {
                this.userData.history.dates.shift();
                this.userData.history.steps.shift();
                this.userData.history.exercise.shift();
            }
        } else {
            this.userData.history.steps[this.userData.history.steps.length - 1] = this.userData.steps_today;
            this.userData.history.exercise[this.userData.history.exercise.length - 1] = this.userData.exercise_minutes;
        }
    }

    updateUI() {
        // Update steps
        document.getElementById('steps-value').textContent = this.userData.steps_today.toLocaleString();
        const stepsProgress = Math.min((this.userData.steps_today / this.userData.steps_goal) * 100, 100);
        document.querySelector('.metric-card .progress-fill').style.width = stepsProgress + '%';
        
        // Update exercise
        document.getElementById('exercise-value').textContent = this.userData.exercise_minutes;
        const exerciseProgress = Math.min((this.userData.exercise_minutes / this.userData.exercise_goal) * 100, 100);
        document.querySelectorAll('.metric-card .progress-fill')[1].style.width = exerciseProgress + '%';
        
        // Update level and XP
        document.getElementById('level').textContent = this.userData.level;
        document.getElementById('streak-days').textContent = this.userData.streak_days;
        const xpProgress = (this.userData.xp / this.userData.xp_to_next) * 100;
        document.querySelector('.xp-fill').style.width = xpProgress + '%';
        document.querySelector('.xp-bar').nextElementSibling.textContent = `${Math.floor(this.userData.xp)}/${this.userData.xp_to_next} XP`;
        
        // Update badges
        this.updateBadges();
        
        // Update charts
        this.updateCharts();
    }

    updateBadges() {
        const badgeGrid = document.querySelector('.badge-grid');
        badgeGrid.innerHTML = '';
        
        this.badges.forEach(badge => {
            const badgeDiv = document.createElement('div');
            badgeDiv.className = `badge ${this.userData.badges.includes(badge.id) ? 'earned' : 'locked'}`;
            badgeDiv.innerHTML = `
                <div class="badge-icon">${badge.icon}</div>
                <div class="badge-name">${badge.name}</div>
            `;
            badgeGrid.appendChild(badgeDiv);
        });
    }

    updateCharts() {
        const stepsBars = document.querySelectorAll('.chart-bar .bar');
        const exerciseBars = document.querySelectorAll('.chart-bar .bar');
        
        if (this.userData.history.steps.length > 0) {
            const maxSteps = Math.max(...this.userData.history.steps);
            const maxExercise = Math.max(...this.userData.history.exercise);
            
            stepsBars.forEach((bar, index) => {
                if (index < this.userData.history.steps.length) {
                    const height = (this.userData.history.steps[index] / maxSteps) * 100;
                    bar.style.height = Math.max(height, 10) + '%';
                }
            });
            
            exerciseBars.forEach((bar, index) => {
                if (index < this.userData.history.exercise.length) {
                    const height = (this.userData.history.exercise[index] / maxExercise) * 100;
                    bar.style.height = Math.max(height, 10) + '%';
                }
            });
        }
    }

    showAchievement(message) {
        document.getElementById('achievement-desc').textContent = message;
        document.getElementById('achievement-popup').style.display = 'block';
    }

    closeAchievement() {
        document.getElementById('achievement-popup').style.display = 'none';
    }

    startWalking() {
        const steps = Math.floor(Math.random() * 200) + 50;
        this.addSteps(steps);
        this.showAchievement(`Walked ${steps} steps! Keep it up!`);
    }

    startExercise(exerciseName) {
        const minutes = Math.floor(Math.random() * 15) + 5;
        this.addExercise(minutes);
        this.showAchievement(`${exerciseName} completed! +${minutes} minutes`);
    }

    resetDay() {
        this.userData.steps_today = 0;
        this.userData.exercise_minutes = 0;
        this.userData.daily_goals_completed = 0;
        this.updateUI();
        this.saveUserData();
    }
}

// Initialize app
let app;

function showTab(tabName) {
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.add('hidden'));
    
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => tab.classList.remove('active'));
    
    document.getElementById(tabName).classList.remove('hidden');
    event.target.classList.add('active');
}

function startWalking() {
    app.startWalking();
}

function startExercise(exerciseName) {
    app.startExercise(exerciseName);
}

function closeAchievement() {
    app.closeAchievement();
}

function resetDay() {
    if (confirm('Reset today\'s progress?')) {
        app.resetDay();
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    app = new WalkingApp();
});
