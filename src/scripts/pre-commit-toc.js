#!/usr/bin/env node

/*
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

/*
Runs markdown-toc for pre-commits. This is necessary to handle multiple file
arguments, which markdown-toc does not.

Humans should generally use markdown-toc following the install instructions
from https://github.com/jonschlinkert/markdown-toc.
*/

const mdtoc = require('markdown-toc');
const fs = require('fs');

var error = 0;
const files = process.argv.slice(2);
for (var i = 0; i < files.length; ++i) {
  const file = files[i];
  const oldContent = fs.readFileSync(file).toString();
  var newContent = oldContent;

  // Only process files with the toc indicator.
  if (!oldContent.match(/<!-- toc -->/m)) continue;

  // If there's a case-incorrect toc section, fix casing.
  newContent = newContent.replace(
    /\n## Table of contents\n\n<!-- toc -->\n/im,
    '\n## Table of contents\n\n<!-- toc -->\n'
  );
  if (oldContent != newContent) {
    console.log(`Fixed "Table of contents" header in ${file}`);
  }

  // Ensure the file properly labels the toc.
  if (!newContent.match(/\n## Table of contents\n\n<!-- toc -->\n/m)) {
    error = 1;
    console.log(
      `${file} has a toc without a "Table of contents" header. Use:\n` +
        '  ## Table of contents\n\n  <!-- toc -->\n'
    );
    continue;
  }

  // Do the toc substitution.
  newContent = mdtoc.insert(newContent, { bullets: '-' });

  if (oldContent != newContent) {
    console.log(`Updating ${file}`);
    fs.writeFileSync(file, newContent);
  }
}
process.exit(error);
