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
  window,
  ExtensionContext,
  commands,
  WorkspaceConfiguration,
  TextEditor,
  Range,
  DecorationOptions,
  ThemeColor,
  TextDocumentChangeEvent,
} from 'vscode';

import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
} from 'vscode-languageclient/node';

let client: LanguageClient;

const splitLineNumberDecorationType = window.createTextEditorDecorationType({});

function updateSplitLineNumbers(editor: TextEditor | undefined) {
  if (!editor) {
    return;
  }
  const document = editor.document;
  if (!document.fileName.includes('/testdata/')) {
    return;
  }

  const decorations: DecorationOptions[] = [];

  // Find all splits
  const splitStarts: number[] = [];
  for (let i = 0; i < document.lineCount; i++) {
    if (document.lineAt(i).text.startsWith('// ---')) {
      splitStarts.push(i);
    }
  }

  // Find the maximum split length to ensure consistent width across all splits
  let maxSplitLength = 0;
  for (let s = 0; s < splitStarts.length; s++) {
    const startLine = splitStarts[s];
    const endLine = s + 1 < splitStarts.length ? splitStarts[s + 1] : document.lineCount;
    const splitLength = endLine - startLine - 1;
    if (splitLength > maxSplitLength) {
      maxSplitLength = splitLength;
    }
  }

  const maxDigits = maxSplitLength > 0 ? maxSplitLength.toString().length : 1;
  const widthStr = `${maxDigits}ch`;

  // Iterate through each split
  for (let s = 0; s < splitStarts.length; s++) {
    const startLine = splitStarts[s];
    const endLine = s + 1 < splitStarts.length ? splitStarts[s + 1] : document.lineCount;
    const splitLength = endLine - startLine - 1;
    if (splitLength <= 0) {
      continue;
    }

    for (let i = startLine + 1; i < endLine; i++) {
      const splitLineNum = i - startLine;
      decorations.push({
        range: new Range(i, 0, i, 0),
        renderOptions: {
          before: {
            contentText: splitLineNum.toString(),
            color: new ThemeColor('editorLineNumber.foreground'),
            width: widthStr,
            margin: '0 2ch 0 0',
            textDecoration: 'none; text-align: right;',
          }
        }
      });
    }
  }

  editor.setDecorations(splitLineNumberDecorationType, decorations);
}

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

  // Update split line numbers when the active editor changes
  context.subscriptions.push(
    window.onDidChangeActiveTextEditor((editor: TextEditor | undefined) => {
      updateSplitLineNumbers(editor);
    })
  );

  // Update split line numbers when the document changes
  context.subscriptions.push(
    workspace.onDidChangeTextDocument((event: TextDocumentChangeEvent) => {
      if (window.activeTextEditor && event.document === window.activeTextEditor.document) {
        updateSplitLineNumbers(window.activeTextEditor);
      }
    })
  );

  // Initial update
  if (window.activeTextEditor) {
    updateSplitLineNumbers(window.activeTextEditor);
  }
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
