// Utility functions to enrich an alert with re-identification-system collection data
// Assumptions documented inline; graceful fallback on any failure.

const {getReIdDatabase} = require("./db");
const ALERT_ID_REGEX = /^(.+?)-lc-(entry|exit)-(\d+)$/; // detectionId-lc-direction-segmentIndex

function parseAlertId(alertId) {
  if (!alertId || typeof alertId !== 'string') return null;
  const m = alertId.match(ALERT_ID_REGEX);
  if (!m) return null;
  return {
    detectionId: m[1],
    direction: m[2],
    segmentIndex: parseInt(m[3], 10)
  };
}

function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function getLineCrossingsCollection(db) {
  const primary = process.env.LINE_CROSSINGS_COLLECTION || 'line-crossings';
  // Attempt primary; fallback name variant 'line_crossings'
  try {
    return db.collection(primary);
  } catch (_) {
    return db.collection('line-crossings');
  }
}

async function findLineCrossingDoc(db, parsed, alert, attempts) {
  if (!parsed) return null;
  const {detectionId, direction, segmentIndex} = parsed;
  const peripheralId = alert?.peripheral_id;
  const alertStartTime = alert?.start_time ? new Date(alert.start_time) : null;
  const collection = getLineCrossingsCollection(db);

  const directionKeyCandidates = [
    'metadata.annotations.teknoir.org.linedir', // hierarchical form
    'metadata.annotations.teknoir.org/linedir'  // legacy / representation
  ];
  const lineIdKeyCandidates = [
    'metadata.annotations.teknoir.org.lineid',
    'metadata.annotations.teknoir.org/lineid'
  ];
  const lineIdCandidateValue = `lc-${direction}-${segmentIndex}-segments`;

  // Pre-probe: does a document with this data.id exist at all? (no direction filter)
  const idOnlyProbe = await collection.findOne({'data.id': detectionId});
  const annotationValues = idOnlyProbe ? idOnlyProbe.metadata?.annotations || {} : {};
  attempts && attempts.push({
    strategy: 'id_only_probe',
    query: {'data.id': detectionId},
    found: !!idOnlyProbe,
    annotationKeys: Object.keys(annotationValues),
    annotationValues
  });

  // Iterate direction key candidates
  for (const directionKey of directionKeyCandidates) {
    const baseDirectionQuery = {[directionKey]: direction};

    // Strategy 1: Exact data.id match + direction
    const query1 = {...baseDirectionQuery, 'data.id': detectionId};
    let doc = await collection.findOne(query1);
    attempts && attempts.push({strategy: `exact_match(${directionKey})`, query: query1, found: !!doc});
    if (doc) return doc;

    // Strategy 2: Prefix regex on data.id
    const query2 = {...baseDirectionQuery, 'data.id': {$regex: `^${escapeRegex(detectionId)}`}};
    doc = await collection
      .find(query2)
      .sort({'metadata.timestamp': -1})
      .limit(1)
      .next();
    attempts && attempts.push({strategy: `prefix_regex(${directionKey})`, query: query2, found: !!doc});
    if (doc) return doc;

    // Strategy 3: lineId candidate + peripheral id/name using each lineId key candidate
    const peripheralQueryFragments = [];
    if (peripheralId) {
      peripheralQueryFragments.push({'data.peripheral.id': peripheralId});
      peripheralQueryFragments.push({'data.peripheral.name': peripheralId});
    }
    const peripheralOr = peripheralQueryFragments.length > 0 ? {$or: peripheralQueryFragments} : {};

    for (const lineIdKey of lineIdKeyCandidates) {
      const query3 = {...baseDirectionQuery, [lineIdKey]: lineIdCandidateValue, ...peripheralOr};
      doc = await collection.find(query3)
        .sort({'metadata.timestamp': -1})
        .limit(1)
        .next();
      attempts && attempts.push({
        strategy: `lineid+peripheral(${directionKey},${lineIdKey})`,
        query: query3,
        found: !!doc
      });
      if (doc) return doc;
    }

    // Strategy 4: Recent docs for same direction + peripheral
    if (peripheralId) {
      const query4a = {...baseDirectionQuery, 'data.peripheral.id': peripheralId};
      let recentDocs = await collection.find(query4a)
        .sort({'metadata.timestamp': -1})
        .limit(10)
        .toArray();
      attempts && attempts.push({
        strategy: `recent_by_peripheral_id(${directionKey})`,
        query: query4a,
        candidates: recentDocs.length
      });

      if (recentDocs.length === 0) {
        const query4b = {...baseDirectionQuery, 'data.peripheral.name': peripheralId};
        recentDocs = await collection.find(query4b)
          .sort({'metadata.timestamp': -1})
          .limit(10)
          .toArray();
        attempts && attempts.push({
          strategy: `recent_by_peripheral_name(${directionKey})`,
          query: query4b,
          candidates: recentDocs.length
        });
      }

      if (recentDocs.length > 0 && alertStartTime) {
        recentDocs.sort((a, b) => {
          const ta = new Date(a.metadata?.timestamp || a.data?.timestamp || 0).getTime();
          const tb = new Date(b.metadata?.timestamp || b.data?.timestamp || 0).getTime();
          return Math.abs(ta - alertStartTime.getTime()) - Math.abs(tb - alertStartTime.getTime());
        });
        const docPicked = recentDocs[0];
        attempts && attempts.push({
          strategy: `time_proximity(${directionKey})`,
          picked: docPicked?.data?.id || null
        });
        if (docPicked) return docPicked;
      }
    }

    // Strategy 5: Latest doc for direction with this key
    const query5 = {...baseDirectionQuery};
    doc = await collection.find(query5).sort({'metadata.timestamp': -1}).limit(1).next();
    attempts && attempts.push({strategy: `latest_direction(${directionKey})`, query: query5, found: !!doc});
    if (doc) return doc;
  }

  return null;
}

function buildBurstUrls(doc, mediaBaseUrl) {
  if (!doc || !doc.data) return {burstImages: [], cutoutImage: null};
  const normalize = p => (p || '').replace(/^\/+/, '');
  const burst = Array.isArray(doc.data.files) ? doc.data.files : [];
  const burstImages = burst.map(p => `${mediaBaseUrl}/jpeg/${normalize(p)}`);
  const cutoutImage = doc.data.filename ? `${mediaBaseUrl}/jpeg/${normalize(doc.data.filename)}` : null;
  return {burstImages, cutoutImage};
}

function extractPose(doc) {
  if (!doc || !doc.data) return null;
  const {coords, skeleton, keypoints} = doc.data;
  if (!Array.isArray(coords) || !Array.isArray(skeleton)) return null;
  return {
    coords,
    skeleton,
    keypoints: Array.isArray(keypoints) ? keypoints : []
  };
}

function extractClassifiers(doc) {
  if (!doc || !doc.data || !Array.isArray(doc.data.classifiers)) return [];
  return doc.data.classifiers.map(c => ({label: c.label, score: c.score}));
}

function extractLineInfo(doc) {
  if (!doc || !doc.metadata || !doc.metadata.annotations) return {lineDirection: null, lineId: null};
  const ann = doc.metadata.annotations;
  return {
    lineDirection: ann['teknoir.org/linedir'] || null,
    lineId: ann['teknoir.org/lineid'] || null
  };
}

async function enrichAlert(alert, options = {}) {
  const db = getReIdDatabase();
  const {enrich = true, includeRaw = false, debug = false, includeBoth = false} = options;
  if (!enrich) return alert;
  const logicalAlertId = alert.id || alert.detection_id; // fallback
  const parsed = parseAlertId(logicalAlertId);
  if (!parsed) {
    return {...alert, enrichment: {parsed: null, reason: 'unparseable_alert_id'}};
  }
  const attempts = debug ? [] : null;
  const collection = getLineCrossingsCollection(db);

  // Fetch all docs with matching data.id or metadata.id (limit for safety)
  const idQuery = {$or: [{'data.id': parsed.detectionId}, {'metadata.id': parsed.detectionId}]};
  let docs = await collection.find(idQuery).sort({'metadata.timestamp': -1}).limit(6).toArray();
  attempts && attempts.push({strategy: 'bulk_id_lookup', query: idQuery, count: docs.length});

  if (docs.length === 0) {
    // fallback to expanded strategies (direction-specific) if nothing found
    const expandedDoc = await findLineCrossingDoc(db, parsed, alert, attempts);
    if (!expandedDoc) {
      return {...alert, enrichment: {parsed, reason: 'no_line_crossing_doc_found', attempts}};
    }
    docs = [expandedDoc];
  }

  // Classify by direction value found in annotations
  const byDirection = {};
  for (const d of docs) {
    const ann = d.metadata?.annotations || {};
    const dir = (ann['teknoir.org/linedir'] || ann['teknoir.org.linedir'] || '').toLowerCase();
    if (!byDirection[dir]) byDirection[dir] = [];
    byDirection[dir].push(d);
  }

  // Pick primary doc:
  // 1. If a doc matches parsed.direction exactly, choose newest of that direction.
  // 2. Else use first doc.
  const targetDir = parsed.direction.toLowerCase();
  let primaryDoc = (byDirection[targetDir] && byDirection[targetDir][0]) || docs[0];
  attempts && attempts.push({
    strategy: 'primary_selection',
    parsedDirection: targetDir,
    primaryDirection: (primaryDoc.metadata?.annotations?.['teknoir.org/linedir'] || '').toLowerCase()
  });

  const mediaBaseUrl = process.env.MEDIA_SERVICE_BASE_URL || 'https://teknoir.cloud/victra-poc/media-service/api';
  const build = (doc) => {
    const {burstImages, cutoutImage} = buildBurstUrls(doc, mediaBaseUrl);
    return {
      lineDirection: extractLineInfo(doc).lineDirection,
      lineId: extractLineInfo(doc).lineId,
      classifiers: extractClassifiers(doc),
      pose: extractPose(doc),
      burstImages,
      cutoutImage,
      burst: doc?.data?.burst || [],
      raw: includeRaw ? doc : undefined
    };
  };

  const primaryEnrichment = build(primaryDoc);

  let multiDirections = undefined;
  if (includeBoth) {
    multiDirections = {};
    for (const dirKey of Object.keys(byDirection)) {
      if (!dirKey) continue;
      const docForDir = byDirection[dirKey][0]; // newest already due to sort
      multiDirections[dirKey] = build(docForDir);
    }
  }

  const enrichment = {
    parsed,
    ...primaryEnrichment,
    multiDirections,
    debug: debug ? {searchedFor: parsed, attempts, directionsFound: Object.keys(byDirection)} : undefined
  };

  return {...alert, enrichment};
}

module.exports = {
  parseAlertId,
  enrichAlert
};
