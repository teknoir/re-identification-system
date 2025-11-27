// API client for backend communication

// Determine BASE_URL injected by server (index.html can set window.__BASE_URL__ via inline script)
const BASE_URL = (window.__BASE_URL__ || '').replace(/\/+$/, ''); // remove trailing slash
// API base always under /api relative to BASE_URL
const API_BASE = `${BASE_URL}/api`.replace(/^\/\//, '/'); // avoid double leading slash

const api = {
  // Fetch all alerts with optional filters
  async getAlerts(filters = {}) {
    const params = new URLSearchParams();

    if (filters.page) params.append('page', filters.page);
    if (filters.limit) params.append('limit', filters.limit);
    if (filters.search) params.append('search', filters.search);
    if (filters.startDate) params.append('startDate', filters.startDate);
    if (filters.endDate) params.append('endDate', filters.endDate);
    try {
      const response = await fetch(`${API_BASE}/alerts?${params}`);
      if (!response.ok) throw new Error(`Failed to fetch alerts: ${response.status} ${response.statusText}`);
      return await response.json();
    } catch (err) {
      throw new Error(`Network error fetching alerts: ${err.message}`);
    }
  },

  // Fetch a specific alert by ID (with optional enrichment)
  async getAlert(id, options = {}) {
    const params = new URLSearchParams();
    if (options.enrich === false) params.append('enrich', '0');
    if (options.includeRaw) params.append('raw', '1');
    const qs = params.toString();
    try {
      const response = await fetch(`${API_BASE}/alerts/${id}${qs ? `?${qs}` : ''}`);
      if (!response.ok) throw new Error(`Failed to fetch alert: ${response.status} ${response.statusText}`);
      return await response.json();
    } catch (err) {
      throw new Error(`Network error fetching alert: ${err.message}`);
    }
  },

  // Check health status
  async healthCheck() {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) {
      throw new Error(`Health check failed: ${response.statusText}`);
    }
    return response.json();
  },

  // Fetch metadata from mediaservice
  async getMetadata(url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch metadata: ${response.statusText}`);
    }
    return response.json();
  },

  // Fetch burst previews with filters
  async getBursts(filters = {}) {
    const params = new URLSearchParams();
    if (filters.date) params.append('date', filters.date);
    if (filters.direction && filters.direction !== 'both') params.append('direction', filters.direction);
    if (filters.camera) params.append('camera', filters.camera);
    if (filters.limit) params.append('limit', filters.limit);
    const qs = params.toString();
    try {
      const response = await fetch(`${API_BASE}/bursts${qs ? `?${qs}` : ''}`);
      if (!response.ok) throw new Error(`Failed to fetch bursts: ${response.status} ${response.statusText}`);
      return await response.json();
    } catch (err) {
      throw new Error(`Network error fetching bursts: ${err.message}`);
    }
  }
};
