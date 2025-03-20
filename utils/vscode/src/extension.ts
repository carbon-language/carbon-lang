/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/*
 * This is the main launcher for the LSP extension.
 */

import {
  workspace,
  ExtensionContext,
  commands,
  WorkspaceConfiguration,
} from 'vscode';

import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

/**
 * Splits a CLI-style quoted string.
 */
function splitQuotedString(argsString: string): string[] {
  const args: string[] = [];
  let arg = '';
  // Track whether there's an arg to handle `""` and similar.
  let hasArg = false;
  // Whether this is in a quote-delimited section.
  let inSingleQuotes = false;
  let inDoubleQuotes = false;
  // Whether this is a `\`-escaped character.
  let inEscape = false;

  for (const char of argsString) {
    // While spaces can appear in arguments, they can only be an argument in
    // combination with other characters.
    hasArg = hasArg || char != ' ';

    if (inEscape) {
      // After an escape, directly append the character.
      arg += char;
      inEscape = false;
      continue;
    }
    switch (char) {
      case '\\':
        // First character of an escape.
        inEscape = true;
        continue;
      case "'":
        if (!inDoubleQuotes) {
          // Single-quoted section.
          inSingleQuotes = !inSingleQuotes;
          continue;
        }
        break;
      case '"':
        if (!inSingleQuotes) {
          // Double-quoted section.
          inDoubleQuotes = !inDoubleQuotes;
          continue;
        }
        break;
      case ' ':
        if (!inSingleQuotes && !inDoubleQuotes) {
          // Space between arguments (but possibly multiple spaces).
          if (hasArg) {
            args.push(arg);
            arg = '';
            hasArg = false;
          }
          continue;
        }
        break;
    }
    arg += char;
  }

  // Finish any pending argument.
  if (hasArg) {
    args.push(arg);
  }

  return args;
}

/**
 * Combines the `language-server` command with args from settings.
 */
function buildServerArgs(settings: WorkspaceConfiguration): string[] {
  const result: string[] = [];
  result.push(
    ...splitQuotedString(settings.get('carbonServerCommandArgs', ''))
  );
  result.push('language-server');
  result.push(
    ...splitQuotedString(settings.get('carbonServerSubcommandArgs', ''))
  );
  return result;
}

export function activate(context: ExtensionContext) {
  const settings = workspace.getConfiguration('carbon');

  const serverOptions: ServerOptions = {
    // The Carbon server can be configured, but we try to use bazel output as a
    // fallback.
    command: settings.get(
      'carbonPath',
      context.asAbsolutePath('./bazel-bin/toolchain/carbon')
    ),
    args: buildServerArgs(settings),
  };

  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ language: 'carbon' }],
  };

  // Create and start the client.
  client = new LanguageClient(
    'carbonLanguageServer',
    'Carbon Language Server',
    serverOptions,
    clientOptions
  );
  client.start();

  context.subscriptions.push(
    commands.registerCommand('carbon.lsp.restart', () => {
      client.restart();
    })
  );
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
