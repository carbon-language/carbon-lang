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
  let inSingleQuotes = false;
  let inDoubleQuotes = false;
  let escaped = false;

  for (const char of argsString) {
    if (escaped) {
      arg += char;
      escaped = false;
      continue;
    }
    switch (char) {
      case '\\':
        escaped = true;
        continue;
      case "'":
        if (!inDoubleQuotes) {
          inSingleQuotes = !inSingleQuotes;
          continue;
        }
        break;
      case '"':
        if (!inSingleQuotes) {
          inDoubleQuotes = !inDoubleQuotes;
          continue;
        }
        break;
      case ' ':
        if (!inSingleQuotes && !inDoubleQuotes) {
          args.push(arg);
          arg = '';
        }
        break;
    }
    arg += char;
  }

  if (arg.length > 0) {
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
