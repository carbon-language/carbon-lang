/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/*
 * Tokenizes the Carbon files named on stdin, one path per line, with the real
 * VS Code engine, and prints a digest of the scope stack of every character,
 * one line per file. `conformance_test.py` compares this against the same
 * digest from `tmlanguage.py`. A mismatch means `re` and Oniguruma no longer
 * agree on this grammar.
 */

import fs from 'node:fs';
import crypto from 'node:crypto';
import vsctm from 'vscode-textmate';
import oniguruma from 'vscode-oniguruma';

const grammarPath = process.argv[2];
const files = fs.readFileSync(0, 'utf8').split('\n').filter(Boolean);

const wasm = new URL(import.meta.resolve('vscode-oniguruma/release/onig.wasm'));
await oniguruma.loadWASM(fs.readFileSync(wasm));
const registry = new vsctm.Registry({ onigLib: Promise.resolve(oniguruma) });
// tmlanguage.py ignores includes of other grammars, so stub out `source.cpp`
// here to match.
await registry.addGrammar({ scopeName: 'source.cpp', patterns: [] });
const grammar = await registry.addGrammar(
  JSON.parse(fs.readFileSync(grammarPath, 'utf8'))
);

const out = [];
for (const file of files) {
  // Python translates line endings on read, so match that before splitting or
  // a CRLF file reports a false mismatch.
  const lines = fs
    .readFileSync(file, 'utf8')
    .replace(/\r\n?/g, '\n')
    .split('\n');
  const hash = crypto.createHash('sha256');
  let stack = vsctm.INITIAL;
  for (const line of lines) {
    const result = grammar.tokenizeLine(line, stack);
    const perChar = new Array(line.length).fill('-');
    for (const token of result.tokens) {
      const end = Math.min(token.endIndex, line.length);
      for (let i = token.startIndex; i < end; i++) {
        perChar[i] = token.scopes.join('/');
      }
    }
    hash.update(perChar.join(',') + ';');
    stack = result.ruleStack;
  }
  out.push(hash.digest('hex') + ' ' + file);
}
process.stdout.write(out.join('\n') + '\n');
