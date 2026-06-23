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
  if (document.languageId !== 'carbon-testdata') {
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

// Get an SVG image with a diagonal "C++" logo.
const getCppSvgBase64 = (color: string, opacity: number) => {
  const svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="2.1 2.3 15.6 15.6">',
    `<g fill="none" stroke-linecap="square" stroke="${color}" `,
    `opacity="${opacity}" transform="rotate(-45 10 10)">`,
    `<circle cx="4.45" cy="10" r="2.7" stroke-width="1.8" `,
    `stroke-dasharray="10.76 6.2" stroke-dashoffset="-3.1" />`,
    `<g stroke-width="1.8">`,
    `<line x1="10.05" y1="8.1" x2="10.05" y2="11.9" />`,
    `<line x1="8.15" y1="10" x2="11.95" y2="10" />`,
    `<line x1="16.35" y1="8.1" x2="16.35" y2="11.9" />`,
    `<line x1="14.45" y1="10" x2="18.25" y2="10" />`,
    `</g></g></svg>`,
  ].join('');
  return Buffer.from(svg).toString('base64');
};

const lightSvgBase64 = getCppSvgBase64('rgb(0,0,0)', 0.07);
const darkSvgBase64 = getCppSvgBase64('rgb(255,255,255)', 0.06);

// Create a decoration type for C++ code embedded in Carbon.
const createCppDecorationType = (isWholeLine: boolean) => {
  // We can't directly set a background image on a decoration, but we can set it
  // via a CSS escape in the border property.
  const getBorder = (base64: string) =>
    /*border:*/ `none; ` +
    `background-image: url('data:image/svg+xml;base64,${base64}'); ` +
    `background-repeat: repeat; ` +
    `background-size: auto 100%;`;
  return window.createTextEditorDecorationType({
    light: { border: getBorder(lightSvgBase64) },
    dark: { border: getBorder(darkSvgBase64) },
    isWholeLine,
  });
};

const cppBlockDecorationType = createCppDecorationType(true);
const cppInlineDecorationType = createCppDecorationType(false);

function updateCppInlineDecorations(editor: TextEditor | undefined) {
  if (!editor) {
    return;
  }
  const document = editor.document;
  if (document.languageId !== 'carbon' &&
      document.languageId !== 'carbon-testdata') {
    return;
  }

  const text = document.getText();
  const blockDecorations: DecorationOptions[] = [];
  const inlineDecorations: DecorationOptions[] = [];

  // 1. Triple-quoted blocks
  const tripleRegex = /(?:import\s+Cpp\s+inline|inline\s+Cpp)\s*('''[^\s'#]*\n?([\s\S]*?)''')\s*;/g;
  let match;
  while ((match = tripleRegex.exec(text)) !== null) {
    const content = match[2];
    if (content) {
      const startOffset = match.index + match[0].indexOf(content);
      let endOffset = startOffset + content.length;

      // If the content ends with a newline, exclude it from the range so that the
      // line containing the closing quotes is not highlighted.
      if (content.endsWith('\r\n')) {
        endOffset -= 2;
      } else if (content.endsWith('\n')) {
        endOffset--;
      }

      blockDecorations.push({
        range: new Range(
          document.positionAt(startOffset),
          document.positionAt(endOffset)
        )
      });
    }
  }

  // 2. Double-quoted inline strings
  const doubleRegex = /(?:import\s+Cpp\s+inline|inline\s+Cpp)\s*("((?:[^"\\]|\\.)*)")\s*;/g;
  while ((match = doubleRegex.exec(text)) !== null) {
    const content = match[2];
    if (content) {
      const startOffset = match.index + match[0].indexOf(content);
      const endOffset = startOffset + content.length;
      inlineDecorations.push({
        range: new Range(
          document.positionAt(startOffset),
          document.positionAt(endOffset)
        )
      });
    }
  }

  editor.setDecorations(cppBlockDecorationType, blockDecorations);
  editor.setDecorations(cppInlineDecorationType, inlineDecorations);
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
    documentSelector: [
      { language: 'carbon' },
      { language: 'carbon-testdata' },
    ],
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
      updateCppInlineDecorations(editor);
    })
  );

  // Update split line numbers when the document changes
  context.subscriptions.push(
    workspace.onDidChangeTextDocument((event: TextDocumentChangeEvent) => {
      if (window.activeTextEditor && event.document === window.activeTextEditor.document) {
        updateSplitLineNumbers(window.activeTextEditor);
        updateCppInlineDecorations(window.activeTextEditor);
      }
    })
  );

  // Initial update
  if (window.activeTextEditor) {
    updateSplitLineNumbers(window.activeTextEditor);
    updateCppInlineDecorations(window.activeTextEditor);
  }
}

export function deactivate(): Thenable<void> | undefined {
  if (!client) {
    return undefined;
  }
  return client.stop();
}
