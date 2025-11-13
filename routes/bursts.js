const express = require('express');
const { getReIdDatabase } = require('../utils/db');

const router = express.Router();

function getLineCrossingsCollection(db) {
  const primary = process.env.LINE_CROSSINGS_COLLECTION || 'line_crossings';
  try {
    return db.collection(primary);
  } catch (_) {
    return db.collection('line_crossings');
  }
}

function normalizeMediaPath(path) {
  return (path || '').replace(/^\/+/, '');
}

function resolveNested(obj, pathParts) {
  if (!obj) return null;
  let current = obj;
  for (const part of pathParts) {
    if (current && typeof current === 'object' && Object.prototype.hasOwnProperty.call(current, part)) {
      current = current[part];
    } else {
      return null;
    }
  }
  return typeof current === 'string' ? current : null;
}

router.get('/', async (req, res) => {
  try {
    const db = getReIdDatabase();
    const collection = getLineCrossingsCollection(db);
    const limit = Math.min(parseInt(req.query.limit, 10) || 60, 120);
    const mediaBaseUrl = process.env.MEDIA_SERVICE_BASE_URL || 'https://teknoir.cloud/victra-poc/media-service/api';
    const andFilters = [];

    const { date, direction = 'both', camera } = req.query;

    if (date) {
      const start = new Date(`${date}T00:00:00.000Z`);
      if (Number.isNaN(start.getTime())) {
        return res.status(400).json({ error: 'Invalid date parameter' });
      }
      const end = new Date(start);
      end.setUTCDate(end.getUTCDate() + 1);

      const startIso = start.toISOString();
      const endIso = end.toISOString();

      andFilters.push({
        $or: [
          { 'metadata.timestamp': { $gte: startIso, $lt: endIso } },
          { 'metadata.timestamp': { $gte: start, $lt: end } },
          { 'data.timestamp': { $gte: startIso, $lt: endIso } },
          { 'data.timestamp': { $gte: start, $lt: end } },
          { 'metadata.timestamp': { $regex: `^${date}` } },
          { 'data.timestamp': { $regex: `^${date}` } }
        ]
      });
    }

    const requestedDirection = (direction || 'both').toLowerCase();

    if (camera) {
      const cameraRegex = new RegExp(camera, 'i');
      andFilters.push({
        $or: [
          { 'data.peripheral.id': cameraRegex },
          { 'data.peripheral.name': cameraRegex }
        ]
      });
    }

    const filter = andFilters.length ? { $and: andFilters } : {};

    const docs = await collection
      .find(filter)
      .sort({ 'metadata.timestamp': -1 })
      .limit(limit)
      .toArray();

    const bursts = docs.map((doc) => {
      const burst = Array.isArray(doc.data?.burst) ? doc.data.burst : [];
      const burstImagesFull = burst.map((p) => `${mediaBaseUrl}/jpeg/${normalizeMediaPath(p)}`);
      const directionValue =
        doc.metadata?.annotations?.['teknoir.org/linedir'] ||
        doc.metadata?.annotations?.['teknoir.org.linedir'] ||
        resolveNested(doc.metadata?.annotations, ['teknoir', 'org', 'linedir']) ||
        doc.metadata?.annotations?.linedir ||
        null;

      const timestamp = doc.metadata?.timestamp || doc.data?.timestamp || null;
      const cutoutImage = doc.data?.filename ? `${mediaBaseUrl}/jpeg/${normalizeMediaPath(doc.data.filename)}` : null;
      const previewImages = burstImagesFull.slice(0, 12);
      if (cutoutImage) {
        const existingIndex = previewImages.indexOf(cutoutImage);
        if (existingIndex > -1) previewImages.splice(existingIndex, 1);
        previewImages.unshift(cutoutImage);
      }

      return {
        id: doc._id,
        detectionId: doc.data?.id || null,
        burstCount: burst.length,
        burstImages: previewImages,
        cutoutImage,
        peripheral: {
          id: doc.data?.peripheral?.id || null,
          name: doc.data?.peripheral?.name || null
        },
        direction: directionValue,
        timestamp
      };
    });

    const filteredBursts = requestedDirection === 'both'
      ? bursts
      : bursts.filter((b) => typeof b.direction === 'string' && b.direction.toLowerCase() === requestedDirection);

    if (process.env.NODE_ENV !== 'production') {
      console.log('[Bursts] Query', JSON.stringify(filter));
      console.log('[Bursts] Direction filter', requestedDirection);
      console.log('[Bursts] Matched docs', docs.length);
      console.log('[Bursts] After direction filter', filteredBursts.length);
      if (filteredBursts[0]) {
        console.log('[Bursts] Sample burst direction', filteredBursts[0].direction);
      }
    }

    res.json({
      total: filteredBursts.length,
      bursts: filteredBursts
    });
  } catch (error) {
    console.error('Error fetching bursts:', error);
    res.status(500).json({ error: 'Failed to fetch bursts', message: error.message });
  }
});

module.exports = router;
