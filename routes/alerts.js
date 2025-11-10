const express = require('express');
const { getDatabase } = require('../utils/db');
const { ObjectId } = require('mongodb');
const { enrichAlert } = require('../utils/enrichAlert');

const router = express.Router();

// GET /api/alerts - List all alerts with pagination and search
router.get('/', async (req, res) => {
  try {
    const db = getDatabase();
    const collection = db.collection('alerts');

    // Pagination
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    // Base filter - only line_crossing alerts
    let baseFilter = { type: 'line_crossing' };
    let filter = { ...baseFilter };

    // Search filter on alert ID (the 'id' field, not '_id')
    if (req.query.search) {
      filter.id = { $regex: req.query.search, $options: 'i' }; // case-insensitive search
    }

    // Determine storage type of start_time once (Date vs String)
    const sample = await collection.findOne(baseFilter, { projection: { start_time: 1 } });
    const startTimeIsDate = sample && sample.start_time instanceof Date;

    let appliedDateFilter = false;
    if (req.query.startDate || req.query.endDate) {
      const startDateValue = req.query.startDate ? new Date(req.query.startDate) : null;
      const endDateValue = req.query.endDate ? new Date(req.query.endDate) : null;

      if (startTimeIsDate) {
        // Normal direct date comparison
        filter.start_time = {};
        if (startDateValue) filter.start_time.$gte = startDateValue;
        if (endDateValue) filter.start_time.$lte = endDateValue;
      } else {
        // Use $expr with $dateFromString for string stored timestamps (assuming ISO format strings)
        const exprConditions = [];
        if (startDateValue) {
          exprConditions.push({
            $gte: [
              { $dateFromString: { dateString: '$start_time' } },
              startDateValue
            ]
          });
        }
        if (endDateValue) {
          exprConditions.push({
            $lte: [
              { $dateFromString: { dateString: '$start_time' } },
              endDateValue
            ]
          });
        }
        if (exprConditions.length > 0) {
          filter.$expr = { $and: exprConditions };
        }
      }
      appliedDateFilter = true;
    }

    // Get total count for pagination (after filters)
    const total = await collection.countDocuments(filter);

    // If date filter applied and no results, but there are alerts without date filter, surface warning
    let dateFilterWarning = false;
    if (appliedDateFilter && total === 0) {
      const totalWithoutDate = await collection.countDocuments(baseFilter);
      if (totalWithoutDate > 0) {
        dateFilterWarning = true;
      }
    }

    // Fetch alerts - sorted by start_time descending (latest first)
    const alerts = await collection
      .find(filter)
      .sort({ start_time: -1 })
      .skip(skip)
      .limit(limit)
      .toArray();

    // Transform alerts to include direct media service URLs
    const mediaBaseUrl = process.env.MEDIA_SERVICE_BASE_URL || 'https://teknoir.cloud/victra-poc/media-service/api';
    const transformedAlerts = alerts.map(alert => ({
      ...alert,
      imageUrl: alert.video_snapshot ? `${mediaBaseUrl}/jpeg/${alert.video_snapshot}` : null,
      videoUrl: alert.video_url ? `${mediaBaseUrl}/mp4/${alert.video_url}` : null,
      metadataUrl: alert.annotations_url ? `${mediaBaseUrl}/json/${alert.annotations_url}` : null
    }));

    res.json({
      alerts: transformedAlerts,
      pagination: {
        page,
        limit,
        total,
        pages: Math.ceil(total / limit)
      },
      meta: {
        dateFilterApplied: appliedDateFilter,
        dateFilterWarning,
        startTimeType: startTimeIsDate ? 'Date' : 'String'
      }
    });
  } catch (error) {
    console.error('Error fetching alerts:', error);
    res.status(500).json({ error: 'Failed to fetch alerts', message: error.message });
  }
});

// GET /api/alerts/:id - Get specific alert details
router.get('/:id', async (req, res) => {
  try {
    const db = getDatabase();
    const collection = db.collection('alerts');

    const alert = await collection.findOne({ _id: new ObjectId(req.params.id) });

    if (!alert) {
      return res.status(404).json({ error: 'Alert not found' });
    }

    // Transform alert to include direct media service URLs
    const mediaBaseUrl = process.env.MEDIA_SERVICE_BASE_URL || 'https://teknoir.cloud/victra-poc/media-service/api';
    let transformedAlert = {
      ...alert,
      imageUrl: alert.video_snapshot ? `${mediaBaseUrl}/jpeg/${alert.video_snapshot}` : null,
      videoUrl: alert.video_url ? `${mediaBaseUrl}/mp4/${alert.video_url}` : null,
      metadataUrl: alert.annotations_url ? `${mediaBaseUrl}/json/${alert.annotations_url}` : null
    };

    // Enrichment (default on unless enrich=0)
    const enrichFlag = req.query.enrich !== '0';
    const includeRaw = req.query.raw === '1' || req.query.debug === '1';
    const debugFlag = req.query.debug === '1';
    const includeBoth = req.query.includeBoth === '1';

    if (enrichFlag) {
      try {
        transformedAlert = await enrichAlert(transformedAlert, db, { enrich: true, includeRaw, debug: debugFlag, includeBoth });
        if (debugFlag && transformedAlert.enrichment?.debug?.attempts) {
          transformedAlert.enrichment.debug.prettyPrintedAttempts = transformedAlert.enrichment.debug.attempts.map(a => ({
            strategy: a.strategy,
            found: a.found,
            candidates: a.candidates,
            picked: a.picked,
            count: a.count,
            query: a.query ? JSON.parse(JSON.stringify(a.query)) : undefined
          }));
        }
      } catch (e) {
        transformedAlert.enrichment = { error: e.message };
      }
    }

    res.json(transformedAlert);
  } catch (error) {
    console.error('Error fetching alert:', error);
    res.status(500).json({ error: 'Failed to fetch alert', message: error.message });
  }
});

module.exports = router;
