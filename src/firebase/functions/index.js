// Part of the Carbon Language, under the Apache License v2.0 with LLVM
// Exceptions.
// See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

'use strict';

const functions = require('firebase-functions');
const admin = require('firebase-admin');
admin.initializeApp({
  authDomain: "carbon-lang.dev",
});
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

// Checks for a session cookie. If present, the content will be added as
// req.user.
const validateFirebaseIdToken = async (req, res, next) => {
  if (!req.cookies || !req.cookies.__session) {
    // Not logged in.
    res.redirect(302, "/login.html");
    return;
  }

  // Validate the cookie, and get user info from it.
  const idToken = req.cookies.__session;
  try {
    req.user = await admin.auth().verifyIdToken(idToken);
  } catch (error) {
    // Invalid login, use logout to clear it.
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
      return;
    }
    // No access, force logout.
    res.redirect(302, "/logout.html");
  } catch (err) {
    // Not a member, force logout.
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
