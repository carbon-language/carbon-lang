// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

'use strict';

const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp();
const cookieParser = require('cookie-parser')();
const cors = require('cors')({origin: true});

const path = require('path');
const express = require('express');
const app = express();

const {SecretManagerServiceClient} = require('@google-cloud/secret-manager');
const secrets = new SecretManagerServiceClient();

const { Octokit } = require('@octokit/rest');

const {Storage} = require('@google-cloud/storage');
const gcs = new Storage();

app.post('/loginSession', (req, res) => {
  // Get the ID token passed and the CSRF token.
  const idToken = req.body.idToken.toString();
  // Create a 14-day session (the limit).
  const expiresIn = 14 * 24 * 60 * 60 * 1000;
  admin.auth().createSessionCookie(idToken, {expiresIn})
    .then((sessionCookie) => {
      console.log("loginSession: Created session cookie");
      const options = {maxAge: expiresIn, httpOnly: true, secure: true};
      res.cookie('session', sessionCookie, options);
      res.end(JSON.stringify({status: 'success'}));
    }, error => {
      console.log("loginSession: Rejecting due to: " + error);
      res.redirect(302, "/logout.html");
    });
});

// Checks for a session cookie. If present, the content will be added as
// req.user.
const validateFirebaseIdToken = async (req, res, next) => {
  if (!req.cookies || !req.cookies.session) {
    // Not logged in.
    console.log("validateFirebaseIdToken: No cookie");
    res.redirect(302, "/login.html");
    return;
  }

  try {
    req.user =
      await admin.auth().verifySessionCookie(req.cookies.session, true);
  } catch (error) {
    // Invalid login, use logout to clear it.
    console.log("validateFirebaseIdToken: Invalid session: " + error);
    res.redirect(302, "/logout.html");
    return;
  }

  // Load the GitHub auth token. The associated secret is attached to the
  // CarbonLangInfra GitHub account.
  const [secret] = await secrets.accessSecretVersion({
    name: 'projects/985662022432/secrets/github-org-lookup-token-for-www/versions/latest',
  });
  const octokit = new Octokit({
    auth: secret.payload.data.toString('utf8'),
  });

  // Translate the GitHub ID to a username.
  const wantId = req.user.firebase.identities["github.com"][0];
  const { data: ghUser } = await octokit.users.list({
    since: wantId - 1,
    per_page: 1,
  });
  if (ghUser.length < 1 || ghUser[0].id != wantId) {
    // An issue with the GitHub ID.
    console.log("validateFirebaseIdToken: GitHub ID issue: " + wantId);
    res.redirect(302, "/logout.html");
  }
  const username = ghUser[0].login;

  // Validate that the GitHub user is in carbon-language.
  try {
    const { data: member } = await octokit.orgs.getMembership({
      org: 'carbon-language',
      username: username,
    });
    if (member && member.state == 'active' && member.user.id == wantId) {
      next();
      console.log("validateFirebaseIdToken: Accepted");
      return;
    }
    // No access, force logout.
    console.log("validateFirebaseIdToken: Not an active member: " + wantId);
    res.redirect(302, "/logout.html");
  } catch (err) {
    // Not a member, force logout.
    console.log("validateFirebaseIdToken: Not a member: " + wantId);
    res.redirect(302, "/logout.html");
  }
};

app.use(cors);
app.use(cookieParser);
app.use(validateFirebaseIdToken);

app.get('*', (req, res) => {
  // Remove the prefix /, and default to index.html.
  var file = req.path.replace(/^(\/)/, "");
  if (file === "") {
    file = "index.html";
  }
  res.type(path.extname(file));
  // Serve the requested data from the carbon-lang bucket.
  const bucket = gcs.bucket("gs://www.carbon-lang.dev");
  const stream = bucket.file(file).createReadStream();
  //stream.on('error', function(err) { res.status(404).send(err.message); });
  stream.on('error', function(err) {
    console.log(err.message);
    res.status(404).send("Not found");
  });
  stream.pipe(res);
});

exports.site = functions.https.onRequest(app);
