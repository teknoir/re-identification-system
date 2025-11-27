const {MongoClient} = require('mongodb');

let historianDB = null;
let reIdDB = null;

let historianClient = null;
let reIdClient = null;


async function connectToHistorianDatabase() {
    if (historianDB) {
        return historianDB;
    }

    const uri = process.env.HISTORIAN_MONGODB_URI;
    if (!uri) {
        throw new Error('HISTORIAN_MONGODB_URI environment variable is not set');
    }

    try {
        historianClient = new MongoClient(uri);
        await historianClient.connect();

        // Extract database name from URI or use default
        const dbName = 'historian';
        historianDB = historianClient.db(dbName);

        console.log(`Connected to Historian MongoDB`);
        console.log(`Using database: ${dbName}`);
        return historianDB;
    } catch (error) {
        console.error('MongoDB connection error:', error);
        throw error;
    }
}

async function connectToReIdDatabase() {
    if (reIdDB) {
        return reIdDB;
    }

    const uri = process.env.REID_MONGODB_URI;
    if (!uri) {
        throw new Error('REID_MONGODB_URI environment variable is not set');
    }

    try {
        reIdClient = new MongoClient(uri);
        await reIdClient.connect();

        // Extract database name from URI or use default
        const dbName = 'historian';
        reIdDB = reIdClient.db(dbName);

        console.log(`Connected to ReId MongoDB`);
        console.log(`Using database: ${dbName}`);
        return reIdDB;
    } catch (error) {
        console.error('MongoDB connection error:', error);
        throw error;
    }
}

function getHistorianDatabase() {
    if (!historianDB) {
        throw new Error('Database not connected. Call connectToHistorianDatabase first.');
    }
    return historianDB;
}

function getReIdDatabase() {
    if (!reIdDB) {
        throw new Error('Database not connected. Call connectToReIdDatabase first.');
    }
    return reIdDB;
}

async function closeHistorianDatabase() {
    if (historianClient) {
        await historianClient.close();
        historianDB = null;
        historianClient = null;
    }
}

async function closeReIdDatabase() {
    if (reIdClient) {
        await reIdClient.close();
        reIdDB = null;
        reIdClient = null;
    }
}

module.exports = {
    connectToHistorianDatabase,
    connectToReIdDatabase,
    getHistorianDatabase,
    getReIdDatabase,
    closeHistorianDatabase,
    closeReIdDatabase
};

