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

const files = process.argv.slice(2);
for (var i = 0; i < files.length; ++i) {
  const oldContent = fs.readFileSync(files[i]).toString();
  const newContent = mdtoc.insert(oldContent, { bullets: '-' });
  if (oldContent != newContent) {
    console.log(`Updating ${files[i]}`);
    fs.writeFileSync(files[i], newContent);
  }
}
