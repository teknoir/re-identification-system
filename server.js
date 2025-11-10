require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const { connectToDatabase } = require('./utils/db');
const alertsRouter = require('./routes/alerts');
const burstsRouter = require('./routes/bursts');

const app = express();
const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || '0.0.0.0'; // bind to all interfaces by default

// Normalize BASE_URL provided in env (examples: '', '/', '/app', 'app/' -> '', '', '/app', '/app')
function normalizeBaseUrl(val) {
  if (!val) return '';
  if (val === '/') return '';
  if (!val.startsWith('/')) val = '/' + val; // ensure leading slash
  return val.replace(/\/+$/, ''); // remove trailing slash(es)
}
const BASE_URL = normalizeBaseUrl(process.env.BASE_URL);

// Middleware
app.use(cors());
app.use(express.json());

// Create a router scoped to BASE_URL so all content (static + API) lives under that path
const baseRouter = express.Router();

// Static assets (index.html, css, js, etc.) served beneath BASE_URL
baseRouter.use(express.static(path.join(__dirname, 'public')));

// Log all requests (scoped under BASE_URL)
baseRouter.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${BASE_URL}${req.path}`);
  next();
});

// API Routes under BASE_URL (/BASE_URL/api/...)
baseRouter.use('/api/alerts', alertsRouter);
baseRouter.use('/api/bursts', burstsRouter);

baseRouter.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString(), baseUrl: BASE_URL || '/' });
});

// Mount the base router at BASE_URL ('' mounts at root when BASE_URL empty)
app.use(BASE_URL, baseRouter);

// Optional redirect from root to BASE_URL if BASE_URL is non-empty
if (BASE_URL) {
  app.get('/', (req, res) => {
    // Preserve query string if any
    const qs = req.url.includes('?') ? req.url.slice(req.url.indexOf('?')) : '';
    res.redirect(`${BASE_URL}/${qs}`);
  });
}

// Start server
async function startServer() {
  try {
    await connectToDatabase();
    console.log('Connected to MongoDB');

    app.listen(PORT, HOST, () => {
      console.log(`Server running on http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}${BASE_URL}`);
      console.log(`Listening interface: ${HOST}`);
      console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
      console.log(`BASE_URL: ${BASE_URL || '(root)'}`);
    });
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

startServer();
