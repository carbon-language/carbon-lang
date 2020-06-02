#!/usr/bin/env node

/*
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/*
Updates the list of proposals in proposals/README.md.
*/

const fs = require('fs');
const path = require('path');

// Read the proposal dir relative to this script.
const proposalDir = path.resolve(__dirname, '../../proposals/');

// Identify proposal titles in the file list.
const files = fs.readdirSync(proposalDir);
const proposals = [];
var error = 0;
for (var i = 0; i < files.length; ++i) {
  const file = files[i];
  const fileMatch = file.match(/^p([0-9]{4})\.md/);
  if (!fileMatch) continue;
  const content = fs.readFileSync(path.join(proposalDir, file)).toString();
  const title = content.split('\n')[0];
  const titleMatch = title.match(/^# (.*)$/);
  if (!titleMatch) {
    console.log(`ERROR: ${file} doesn't have a title on the first line.`);
    error = 1;
  }
  proposals.push(`- [${fileMatch[1]} - ${titleMatch[1]}](${file})`);
  const decisionFile = `p${fileMatch[1]}-decision.md`;
  if (fs.existsSync(path.join(proposalDir, decisionFile))) {
    proposals.push(`  - [Decision](${decisionFile})`);
  }
}
if (error) process.exit(error);

// Replace the README content if needed.
const readmePath = path.join(proposalDir, 'README.md');
const oldContent = fs.readFileSync(readmePath).toString();
const proposalsRegex = /(<!-- proposals -->)(?:(?:.|\n)*)(<!-- endproposals -->)/;
if (!oldContent.match(proposalsRegex)) {
  console.log(oldContent);
  console.log(
    'ERROR: proposals/README.md is missing the ' +
      '<!-- proposals --> ... <!-- endproposals --> marker.'
  );
  process.exit(1);
}
const newContent = oldContent.replace(
  proposalsRegex,
  `$1\n\n${proposals.join('\n')}\n\n$2`
);
if (oldContent != newContent) {
  console.log('Updating proposals/README.md');
  fs.writeFileSync(readmePath, newContent);
}
