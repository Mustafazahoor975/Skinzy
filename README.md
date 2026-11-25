# Skinzy

**Skin-care application prototype · React Native, Node.js & Python**

Skinzy explores how a mobile application can bring together patient–doctor interactions, appointment management, messaging and image analysis. The repository contains an Expo application, an Express backend and a separate Flask/PyTorch service.

> **Project status:** Development prototype. The image-analysis component is not a clinically validated diagnostic tool and is not intended to replace professional medical advice.

## Project scope

The source includes workflows for:

- Patient and doctor accounts with role-specific screens.
- Appointment management and patient history.
- Real-time messaging using Socket.io.
- Notifications, feedback and skin-care routines.
- Image-analysis integration through a Python service.

These describe the scope of the codebase, not a claim that every workflow has been validated in a production deployment.

## Architecture

| Component | Responsibilities | Technologies |
| --- | --- | --- |
| Mobile application | Screens, navigation and interaction with application services | React Native, Expo Router, TypeScript, NativeWind |
| Application backend | Accounts, appointments, messaging and persistent application data | Node.js, Express, MongoDB, JWT, Socket.io |
| Image-analysis service | Loading a trained model and processing image-analysis requests | Python, Flask, PyTorch, Torchvision |

## Where to start

- **[Mobile package configuration](./package.json)** — application dependencies and Expo scripts.
- **[Backend](./backend/)** — Express application and backend dependencies.
- **[Model service](./ml/model/api.py)** — Flask service and model-loading logic.
- **[Model resources](./ml/model/)** — training code, model metadata and supporting files.

## Local development notes

The mobile app, backend and model service are separate processes. Running the Expo app alone does not start the other services.

### Mobile application

From the repository root:

```bash
npm install
npm start
```

The project also provides `npm run android`, `npm run ios` and `npm run web` scripts. Configure the application's API addresses for the machine or device used during development.

### Backend

The backend declares Node.js 20 or later. After configuring its database connection and required environment values:

```bash
cd backend
npm install
npm run dev
```

Review the backend configuration before starting it. A working MongoDB connection and the relevant application services are required for dependent workflows.

### Image-analysis service

Review `ml/model/api.py` for Python dependencies, model paths and service configuration. Trained model weights are not included in this repository; a compatible checkpoint is required before inference can work. The model architecture and class ordering must match the checkpoint.

## Current limitations

- Full application setup requires configuration across all three components.
- Model weights must be supplied separately.
- This README does not claim clinical accuracy, production readiness or completed end-to-end testing.
- Test each workflow in your configured environment before relying on it.

## Developer

**Mustafa Zahoor** · Software Engineering student, University of Central Punjab

[GitHub profile](https://github.com/Mustafazahoor975) · [Portfolio](https://portfolio-react-js-tan.vercel.app/) · [LinkedIn](https://www.linkedin.com/in/mustafazahoor/)
