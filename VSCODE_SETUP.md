# 🚀 How to Run Skinzy in VS Code

This guide provides instructions to run the **Skinzy** application in Visual Studio Code. The project consists of three services:
1. **Frontend**: React Native Expo app (Root directory)
2. **Backend**: Express.js API (`backend` directory)
3. **AI/ML Server**: Python Flask server (`ml/model` directory)

---

## 🛠️ Option 1: Run All Together with VS Code Tasks (Recommended)

We have created a `.vscode/tasks.json` file in your workspace. You can run all three components in parallel with a single command inside VS Code:

1. Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on Mac) to open the Command Palette.
2. Type **Tasks: Run Task** and press `Enter`.
3. Select **`🚀 Run All (Backend + AI Server + Frontend)`**.
4. VS Code will open terminals and start the backend, Python server, and Expo CLI concurrently!

---

## 💻 Option 2: Run Separately in Individual VS Code Terminals

If you prefer to start each service manually or need to see individual logs clearly:

### 1. Database Setup
Make sure MongoDB is installed and running locally on your machine.
- Local URI: `mongodb://localhost:27017/skinzy` (Default port `27017`)

### 2. Node.js Backend API
Open a terminal in VS Code:
```bash
# Navigate to backend directory
cd backend

# Install dependencies
npm install

# Run database seeder (Optional, adds mock doctors)
node seed_doctors.js

# Start the development server (runs with nodemon auto-restart)
npm run dev
```
*The backend API will run on `http://localhost:4445` (configured in `backend/.env`).*

### 3. AI/ML Python Server
Open a new terminal tab in VS Code:
```bash
# Navigate to the ML model directory
cd ml/model

# Set up a python virtual environment (Optional but highly recommended)
python -m venv venv
venv\Scripts\activate      # For Windows Command Prompt/PowerShell
# source venv/bin/activate  # For Mac/Linux

# Install required packages
pip install flask flask-cors torch torchvision pillow numpy

# Start the Flask AI server
python api.py
```
*The AI server will run on `http://127.0.0.1:5000`.*

### 4. Frontend App (Expo React Native)
Open a new terminal tab in VS Code:
```bash
# Make sure you are in the root directory of the project
cd "SkinzyApp Copy" # or your root workspace directory

# Install frontend dependencies
npm install

# Start the Expo application
npm start
```
From the Expo menu in your terminal:
- Press **`w`** to open in a web browser.
- Press **`a`** to open in an Android Emulator.
- Press **`i`** to open in an iOS Simulator.
- Scan the QR code with your phone's **Expo Go** app to test on physical iOS/Android devices!

---

## 🔑 Configuration & Variables
- **Backend Environment**: Check/edit [backend/.env](file:///c:/Users/jour/OneDrive%20-%20Punjab%20Group%20of%20Colleges/Documents/SkinzyApp%20Copy/backend/.env)
- **Frontend App Details**: Check [app.json](file:///c:/Users/jour/OneDrive%20-%20Punjab%20Group%20of%20Colleges/Documents/SkinzyApp%20Copy/app.json) and [package.json](file:///c:/Users/jour/OneDrive%20-%20Punjab%20Group%20of%20Colleges/Documents/SkinzyApp%20Copy/package.json)
