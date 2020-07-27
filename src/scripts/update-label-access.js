/*
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/*
Mirrors the carbon-language organization to the contributors-with-label-access
team.

Set GITHUB_AUTH_KEY in your environment to use this script. e.g.:
  GITHUB_AUTH_KEY=abc123 node update-label-access.js

This team exists because we need a team to manage triage access to repos;
GitHub doesn't allow the org to be set to triage access, only read/write.

This does not handle removing ex-members from contributors-with-label-access.
*/

'use strict';

const updateTriageAccess = async () => {
  // The org and team to mirror.
  const org = 'carbon-language';
  const team = 'contributors-with-label-access';
  // Accounts in the org to skip mirroring.
  const ignore = ['CarbonLangInfra', 'google-admin', 'googlebot'];

  // Set up the GitHub API.
  const { Octokit } = require('@octokit/rest');
  const key = process.env.GITHUB_AUTH_KEY;
  if (!key) {
    console.log('Missing GITHUB_AUTH_KEY');
    return;
  }
  const octokit = new Octokit({ auth: key });

  // Load org members.
  var orgMembers = {};
  try {
    const ret = await octokit.paginate(octokit.orgs.listMembers, { org: org });
    for (var i = 0; i < ret.length; ++i) {
      if (ignore.indexOf(ret[i].login) >= 0) continue;
      orgMembers[ret[i].id] = ret[i].login;
    }
  } catch (error) {
    console.log(`org.listMembers failed: ${error}`);
    return;
  }
  console.log(
    `${org} has ${Object.keys(orgMembers).length} members, excluding ${
      ignore.length
    } ignored.`
  );

  // Load team members.
  var teamMembers = new Set();
  try {
    const ret = await octokit.paginate(octokit.teams.listMembersInOrg, {
      org: org,
      team_slug: team,
    });
    for (var i = 0; i < ret.length; ++i) {
      teamMembers[ret[i].id] = ret[i].login;
    }
  } catch (error) {
    console.log(`teams.listMembersInOrg failed: ${error}`);
    return;
  }
  console.log(`${team} has ${Object.keys(teamMembers).length} members.`);

  // Copy members from the org to the team.
  for (const member in orgMembers) {
    if (teamMembers.hasOwnProperty(member)) continue;
    console.log(`Adding ${orgMembers[member]}`);
    octokit.teams.addOrUpdateMembershipForUserInOrg({
      org: org,
      team_slug: team,
      username: orgMembers[member],
    });
  }

  console.log('Done!');
};

updateTriageAccess();
