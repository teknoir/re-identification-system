const { MongoClient } = require('mongodb');

let db = null;
let client = null;

async function connectToDatabase() {
  if (db) {
    return db;
  }

  const uri = process.env.HISTORIAN_MONGODB_URI;
  if (!uri) {
    throw new Error('HISTORIAN_MONGODB_URI environment variable is not set');
  }

  try {
    client = new MongoClient(uri);
    await client.connect();

    // Extract database name from URI or use default
    const dbName = uri.split('/').pop().split('?')[0] || 'historian';
    db = client.db(dbName);

    console.log(`Connected to MongoDB`);
    console.log(`Using database: ${dbName}`);
    return db;
  } catch (error) {
    console.error('MongoDB connection error:', error);
    throw error;
  }
}

function getDatabase() {
  if (!db) {
    throw new Error('Database not connected. Call connectToDatabase first.');
  }
  return db;
}

async function closeDatabase() {
  if (client) {
    await client.close();
    db = null;
    client = null;
  }
}

module.exports = {
  connectToDatabase,
  getDatabase,
  closeDatabase
};

