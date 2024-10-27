/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

const { LanguageClient } = require('vscode-languageclient/node');

function activate(context) {
  const command = './bazel-bin/language_server/language_server';
  const serverOptions = {
    run: { command },
    debug: { command },
  };

  const clientOptions = {
    documentSelector: [{ scheme: 'file', language: 'carbon' }],
  };

  const client = new LanguageClient(
    'languageServer',
    'Language Server for Carbon',
    serverOptions,
    clientOptions
  );

  // stop client on shutdown
  context.subscriptions.push(client.start());
}

function deactivate() {}

module.exports = {
  activate,
  deactivate,
};
