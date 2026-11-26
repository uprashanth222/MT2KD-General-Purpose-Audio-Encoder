# MT2KD web app — frontend

```
frontend/
├── index.html              page structure (HTML only)
├── assets/
│   ├── css/style.css       all styles (theme colours are the --variables at the top)
│   ├── js/config.js        settings: backend address, max recording length
│   ├── js/app.js           all page logic: upload, microphone recording, playback, Demo/Live mode, API calls, results
│   └── manifest.json       web-app metadata (name, colours)
└── README.md
```

- The backend (backend/app.py) serves `index.html` at `/` and this `assets/` folder at `/assets`.
- Open the app at http://localhost:8000 (Live mode). Opening index.html directly from disk shows Demo mode only.
- After editing CSS/JS, refresh with Ctrl+F5 (or raise the `?v=` number in index.html) so the browser loads the new files.
