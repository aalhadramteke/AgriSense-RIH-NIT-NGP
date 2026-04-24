// Additional JS utilities - inline in index.html covers core functionality
// This file for future extensions like service worker, PWA, offline cache

// Market price updates (static for now)
// Risk threshold config
const RISK_THRESHOLDS = {
  ph_low: 5.5,
  n_high: 120,
  rain_drought: 20,
  hum_disease: 90
};

// PWA setup (optional)
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/js/sw.js');
}

// Export for potential modular use
window.AgrisenseUtils = {
  checkRisks, speakRecommendation, CROPS_EMOJI, MARKET_PRICES
};
