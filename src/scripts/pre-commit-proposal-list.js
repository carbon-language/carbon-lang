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

console.log(fs.readdirSync('./proposals/'));
/*
  const oldContent = fs.readFileSync(file).toString();
  if (oldContent != newContent) {
    console.log(`Updating ${file}`);
    fs.writeFileSync(file, newContent);
  }
*/
