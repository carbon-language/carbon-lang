#!/usr/local/bin/node

// Runs markdown-toc for pre-commits. This is necessary to handle multiple file
// arguments, which markdown-toc does not.
//
// Humans should generally use markdown-toc following the install instructions
// from https://github.com/jonschlinkert/markdown-toc.

const mdtoc = require('markdown-toc');
const fs = require('fs');

files = process.argv.slice(2);
for (var i = 0; i < files.length; ++i) {
  const oldContent = fs.readFileSync(files[i]).toString();
  const newContent = mdtoc.insert(oldContent, { bullets: '-' });
  if (oldContent != newContent) {
    console.log(`Updating ${files[i]}`);
    fs.writeFileSync(files[i], newContent);
  }
}
