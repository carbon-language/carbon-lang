// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

'use strict';

const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();
const cookieParser = require('cookie-parser')();
const cors = require('cors')({ origin: true });

const path = require('path');
const express = require('express');
const app = express();

const cache = require('js-cache');
const contentCache = new cache();
const sessionCache = new cache();

const { Storage } = require('@google-cloud/storage');
const gcs = new Storage();

// Validates that the GitHub user associated with the ID token is in the
// 'carbon-language' organization.
const validateIdToken = async (idToken) => {
  const startTime = new Date();
  var result = 'unknown';
  var gitHubId = '<noid>';
  var username = '<nouser>';
  try {
    var user = null;
    try {
      user = await admin.auth().verifyIdToken(idToken);
    } catch (error) {
      return false;
    }
    gitHubId = user.firebase.identities['github.com'][0];

    // The associated secret is attached to the CarbonLangInfra GitHub account.
    const {
      SecretManagerServiceClient,
    } = require('@google-cloud/secret-manager');
    const secrets = new SecretManagerServiceClient();
    const [secret] = await secrets.accessSecretVersion({
      name:
        'projects/985662022432/secrets/github-org-lookup-token-for-www/versions/latest',
    });

    const { Octokit } = require('@octokit/rest');
    const octokit = new Octokit({
      auth: secret.payload.data.toString('utf8'),
    });

    const { data: ghUser } = await octokit.users.list({
      since: gitHubId - 1,
      per_page: 1,
    });
    if (ghUser.length < 1 || ghUser[0].id != gitHubId) {
      result = 'Failed to fetch matching GitHub ID';
      return false;
    }
    username = ghUser[0].login;

    try {
      const { data: member } = await octokit.orgs.getMembership({
        org: 'carbon-language',
        username: username,
      });
      if (member && member.state == 'active' && member.user.id == gitHubId) {
        result = 'Pass';
        return true;
      }
      result = 'Not an active member';
      return false;
    } catch (err) {
      result = 'Not a member';
      return false;
    }

    return false; // Should be unreachable.
  } finally {
    const elapsed = new Date() - startTime;
    console.log(
      `validateIdToken: ${result} (${gitHubId}/${username}; ${elapsed}ms)`
    );
  }
};

// Handles a user logging into a session.
const loginSession = async (req, res) => {
  const startTime = new Date();
  var result = 'unknown';
  try {
    // Get the ID token passed and the CSRF token.
    const idToken = req.body.idToken.toString();
    const isOk = await validateIdToken(idToken);
    if (!isOk) {
      result = 'validateIdToken failed';
      res.redirect(302, '/logout.html');
      return;
    }

    // Create a 14-day session (the limit).
    const expiresIn = 14 * 24 * 60 * 60 * 1000;
    var sessionCookie;
    try {
      sessionCookie = await admin
        .auth()
        .createSessionCookie(idToken, { expiresIn });
    } catch (error) {
      result = `createSessionCookie failed: ${error}`;
      res.redirect(302, '/logout.html');
      return;
    }

    result = 'Pass';
    const options = { maxAge: expiresIn, httpOnly: true, secure: true };
    res.cookie('__session', sessionCookie, options);
    res.end(JSON.stringify({ status: 'success' }));
  } finally {
    const elapsed = new Date() - startTime;
    console.log(`loginSessior: ${result} (${elapsed}ms)`);
  }
};

// Checks for a session cookie.
const validateSessionCookie = async (req, res, next) => {
  const startTime = new Date();
  var result = 'unknown';
  try {
    if (!req.cookies || !req.cookies.__session) {
      // Not logged in.
      result = 'No __session cookie';
      res.redirect(302, '/login.html');
      return;
    }
    const sessionCookie = req.cookies.__session;

    if (sessionCache.get(sessionCookie)) {
      result = 'Pass (cache hit)';
      next();
      return;
    }

    try {
      const user = await admin.auth().verifySessionCookie(sessionCookie, false);
    } catch (error) {
      // Invalid login, use logout to clear it.
      result = `Invalid __session: ${error}`;
      res.redirect(302, '/logout.html');
      return;
    }

    result = 'Pass (cache miss)';
    // Cache positive results for an hour.
    sessionCache.set(sessionCookie, true, 60 * 60 * 1000);
    next();
  } finally {
    const elapsed = new Date() - startTime;
    console.log(
      `validateSessionCookie at ${req.path}: ${result} (${elapsed}ms)`
    );
  }
};

// Handles serving content from GCS.
const serveContent = async (req, res) => {
  const startTime = new Date();
  var result = 'unknown';
  try {
    // Remove the prefix /, and default to index.html.
    var file = req.path.replace(/^(\/)/, '');
    if (file === '') {
      file = 'index.html';
    }

    // Use the extension to determine the MIME type.
    var type = path.extname(file);
    if (type == '') {
      // Treat files with no extension (e.g. LICENSE) as plain text.
      type = 'text/plain';
    }
    res.type(type);

    // Generally allow caching, particularly for css, js, images, etc, but not
    // HTML or extensionless files.
    if (type != '.html' && type != 'text/plain') {
      res.set('Cache-Control', 'public, max-age=31557600');
    }

    // Check cache.
    var cacheHit = 'hit';
    var contents = contentCache.get(file);
    if (!contents) {
      cacheHit = 'miss';
      // Serve the requested data from the carbon-lang bucket.
      const bucket = gcs.bucket('gs://www.carbon-lang.dev');
      var contents;
      try {
        const data = await bucket.file(file).download();
        contents = data[0];
        // Cache content for 15 minutes.
        contentCache.set(file, contents, 5 * 60 * 1000);
      } catch (error) {
        result = `Error: ${error}`;
        res.status(404).send('Not found');
        return;
      }
    }

    result = `Pass (cache ${cacheHit}; ${contents.length} bytes)`;
    res.send(contents);
  } finally {
    const elapsed = new Date() - startTime;
    console.log(`serveContent at ${req.path}: ${result} (${elapsed}ms)`);
  }
};

app.use(cors);
app.use(cookieParser);
app.post('/loginSession', loginSession);
app.use(validateSessionCookie);
app.get('*', serveContent);

exports.site = functions.https.onRequest(app);
