// Simple tests for enrichAlert utilities without external framework
const assert = require('assert');
const { parseAlertId, enrichAlert } = require('../utils/enrichAlert');

// Mock DB and collection behavior
function makeMockDb(docOptions = {}) {
  const storedDoc = docOptions.doc || null;
  return {
    collection(name) {
      assert.strictEqual(name, 're-identification-system');
      return {
        async findOne(query) {
          if (!storedDoc) return null;
          if (query.$or) {
            const match = query.$or.some(cond => cond['data.id'] === storedDoc.data.id || cond['metadata.id'] === storedDoc.metadata?.id);
            if (match) return storedDoc;
          }
          if (query['data.id'] && query['data.id'] === storedDoc.data.id && query['metadata.annotations.teknoir.org/linedir'] === storedDoc.metadata.annotations['teknoir.org/linedir']) {
            return storedDoc;
          }
          return null;
        },
        find(query) {
          let matched = false;
          if (storedDoc) {
            if (query.$or) {
              matched = query.$or.some(cond => cond['data.id'] === storedDoc.data.id || cond['metadata.id'] === storedDoc.metadata?.id);
            } else if (query['metadata.annotations.teknoir.org/linedir']) {
              const directionMatch = query['metadata.annotations.teknoir.org/linedir'] === storedDoc.metadata.annotations['teknoir.org/linedir'];
              if (directionMatch) {
                if (query['data.id'] && query['data.id'].$regex) {
                  const regex = new RegExp(query['data.id'].$regex);
                  matched = regex.test(storedDoc.data.id);
                } else if (query['metadata.annotations.teknoir.org/lineid'] && query['metadata.annotations.teknoir.org/lineid'] === storedDoc.metadata.annotations['teknoir.org/lineid']) {
                  matched = true;
                } else if (query.$or) {
                  matched = query.$or.some(cond => cond['data.peripheral.id'] === storedDoc.data.peripheral?.id || cond['data.peripheral.name'] === storedDoc.data.peripheral?.name);
                } else if (query['data.peripheral.id'] && storedDoc.data.peripheral?.id === query['data.peripheral.id']) {
                  matched = true;
                } else if (query['data.peripheral.name'] && storedDoc.data.peripheral?.name === query['data.peripheral.name']) {
                  matched = true;
                } else if (!query['data.id'] && !query['metadata.annotations.teknoir.org/lineid'] && !query.$or && !query['data.peripheral.id'] && !query['data.peripheral.name']) {
                  matched = true;
                }
              }
            }
          }
          const cursor = {
            sort() { return cursor; },
            limit() { return cursor; },
            async next() { return matched ? storedDoc : null; },
            async toArray() { return matched ? [storedDoc] : []; }
          };
          return cursor;
        }
      };
    }
  };
}

function testParseAlertId() {
  const good = 'nc0009-salefloor-270-155f0f2e-935-lc-exit-0';
  const parsed = parseAlertId(good);
  assert(parsed, 'Parsed object should exist');
  assert.strictEqual(parsed.direction, 'exit');
  assert.strictEqual(parsed.segmentIndex, 0);
  assert.strictEqual(parsed.detectionId, 'nc0009-salefloor-270-155f0f2e-935');

  const bad = 'invalid-format-string';
  assert.strictEqual(parseAlertId(bad), null, 'Should return null for invalid id');
}

async function testEnrichNoParse() {
  const alert = { id: 'bad' };
  const enriched = await enrichAlert(alert, makeMockDb(), { enrich: true });
  assert(enriched.enrichment, 'Enrichment field present');
  assert.strictEqual(enriched.enrichment.reason, 'unparseable_alert_id');
}

async function testEnrichNoDoc() {
  const alert = { id: 'nc0009-salefloor-270-155f0f2e-935-lc-exit-0' };
  const enriched = await enrichAlert(alert, makeMockDb(), { enrich: true });
  assert.strictEqual(enriched.enrichment.reason, 'no_line_crossing_doc_found');
}

async function testEnrichWithDocExact() {
  const alert = { id: 'abc-lc-entry-0' };
  const doc = {
    data: {
      id: 'abc',
      burst: ['media/a.jpg', '/media/b.jpg'],
      filename: 'media/cutout.jpg',
      coords: [[0,0]],
      skeleton: [[0,0]],
      classifiers: [{ label: 'up', score: 0.9 }]
    },
    metadata: { annotations: { 'teknoir.org/linedir': 'entry', 'teknoir.org/lineid': 'lc-entry-0-segments' } }
  };
  const enriched = await enrichAlert(alert, makeMockDb({ doc }), { enrich: true });
  assert(enriched.enrichment, 'Enrichment exists');
  assert.strictEqual(enriched.enrichment.burstImages.length, 2);
  assert.strictEqual(enriched.enrichment.cutoutImage.endsWith('media/cutout.jpg'), true);
  assert.strictEqual(enriched.enrichment.lineDirection, 'entry');
  assert.strictEqual(enriched.enrichment.classifiers[0].label, 'up');
}

async function testEnrichWithDocRegexFallback() {
  const alert = { id: 'prefix-lc-entry-0' };
  const doc = {
    data: { id: 'prefix-xyz-123', burst: [], filename: null, coords: [], skeleton: [], classifiers: [] },
    metadata: { annotations: { 'teknoir.org/linedir': 'entry', 'teknoir.org/lineid': 'lc-entry-0-segments' } }
  };
  const enriched = await enrichAlert(alert, makeMockDb({ doc }), { enrich: true });
  assert(enriched.enrichment, 'Enrichment exists');
  assert.strictEqual(enriched.enrichment.burstImages.length, 0);
  assert.strictEqual(enriched.enrichment.lineDirection, 'entry');
}

async function testEnrichLineIdPeripheralFallback() {
  const alert = { id: 'abc-lc-entry-0', peripheral_id: 'cam-123', start_time: new Date().toISOString() };
  // Doc will not match exact id or prefix; relies on lineId candidate + peripheral
  const doc = {
    data: { id: 'unrelated-id', burst: ['media/x.jpg'], filename: 'media/y.jpg', coords: [], skeleton: [], classifiers: [] , peripheral: { id: 'cam-123'}},
    metadata: { annotations: { 'teknoir.org/linedir': 'entry', 'teknoir.org/lineid': 'lc-entry-0-segments' }, timestamp: new Date().toISOString() }
  };
  const enriched = await enrichAlert(alert, makeMockDb({ doc }), { enrich: true });
  assert.strictEqual(enriched.enrichment.burstImages.length, 1, 'Should pick up burst via fallback');
  assert.strictEqual(enriched.enrichment.lineId, 'lc-entry-0-segments');
}

async function testPrimaryIdLookupNonStrict() {
  const alert = { id: 'xyz-lc-entry-0' };
  const doc = {
    data: { id: 'xyz', burst: [], filename: null, coords: [], skeleton: [], classifiers: [] },
    metadata: { annotations: { 'teknoir.org/linedir': 'EXIT', 'teknoir.org/lineid': 'lc-entry-0-segments' } }
  };
  const enriched = await enrichAlert(alert, makeMockDb({ doc }), { enrich: true });
  assert(enriched.enrichment, 'Enrichment exists via primary id lookup');
  assert.strictEqual(enriched.enrichment.lineDirection, 'EXIT');
}

(async () => {
  try {
    testParseAlertId();
    await testEnrichNoParse();
    await testEnrichNoDoc();
    await testEnrichWithDocExact();
    await testEnrichWithDocRegexFallback();
    await testEnrichLineIdPeripheralFallback();
    await testPrimaryIdLookupNonStrict();
    console.log('All enrichAlert tests passed');
  } catch (e) {
    console.error('Test failure:', e);
    process.exit(1);
  }
})();
