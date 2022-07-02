/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/*
 * Language: Carbon
 * Category: common, system
 * Website: https://github.com/carbon-language/carbon-lang/
 *
 * TODO: local `var` and `let` declarations.
 * TODO: struct literals.
 * TODO: abstract, virtual, impl on methods.
 * TODO: private, protected.
 * TODO: Dedicated package and import highlighting.
 * TODO: Dedicated `impl` highlighting.
 * TODO: `impl forall` pattern highlighting.
 * TODO: Some way to keep keywords here and in other implementations in sync.
 */

/** @type LanguageFn */
export default function (hljs) {
  // Common keywords definition used in various contexts to at least highlight
  // these correctly.
  const KEYWORDS = {
    keyword: [
      'abstract',
      'addr',
      'alias',
      'and',
      'api',
      'as',
      'auto',
      'base',
      'break',
      'case',
      'class',
      'constraint',
      'continue',
      'default',
      'else',
      'extends',
      'external',
      'final',
      'fn',
      'for',
      'forall',
      'friend',
      'if',
      'impl',
      'import',
      'in',
      'interface',
      'is',
      'let',
      'library',
      'match',
      'me',
      'namespace',
      'not',
      'observe',
      'or',
      'override',
      'package',
      'partial',
      'private',
      'protected',
      'return',
      'returned',
      'then',
      '_',
      'var',
      'virtual',
      'where',
      'while',
      'xor',
    ],
    literal: ['false', 'true'],
    type: ['bool'],
    built_in: ['Type', 'As'],
  };

  // Punctuation and operator regex lists that are expanded into the expression
  // context.
  const PUNCTUATION = [/->/, /\./, /:!?/, /;/];
  const OPERATORS = [
    />>=/,
    /<=>/,
    /<<=/,
    /&=/,
    /\^=/,
    /:=/,
    /==/,
    /=>/,
    /!=/,
    />=/,
    />>/,
    /<=/,
    /<>/,
    /<</,
    /<-/,
    /-=/,
    /--/,
    /%=/,
    /\|=/,
    /\+=/,
    /\+\+/,
    /\/=/,
    /\*=/,
    /~=/,
    /&/,
    /\\/,
    /\^/,
    /=/,
    /!/,
    />/,
    /</,
    /-/,
    /%/,
    /\|/,
    /\+/,
    /\?/,
    /\//,
    /\*/,
    /~/,
  ];

  // The core expression patterns.
  const NUMBER_LITERAL = {
    scope: 'number',
    variants: [
      { match: /[1-9][_0-9]*(\.[_0-9]+(e[-+]?[1-9][0-9]*)?)?/ },
      { match: /0x[_0-9A-F]+(\.[_0-9A-F]+(p[-+]?[1-9][0-9]*)?)?/ },
      { match: /0b[_01]+/ },
    ],
  };
  const TYPE_LITERAL = {
    scope: 'type',
    match: /[iuf][1-9][0-9]*/,
  };
  const ESCAPE_SEQUENCE = {
    scope: 'char.escape',
    match: /\\([tnr'"\\0]|0[0-9]|x[0-9A-F]{2}|u\{[0-9A-F]{4,}\})/,
  };
  const BLOCK_STRING_LITERAL = {
    scope: 'string',
    begin: [/"""/, /\S*/, /\n/],
    // TODO: `subLanguages` doesn't support referencing part of the match yet.
    beginScope: {
      2: 'symbol',
    },
    end: /\s*"""/,
    contains: [
      ESCAPE_SEQUENCE,
      {
        scope: 'char.escape',
        match: /\\\n/,
      },
    ],
  };
  const STRING_LITERAL = {
    scope: 'string',
    begin: /"/,
    end: /"/,
    illegal: /\n/,
    contains: [ESCAPE_SEQUENCE],
  };
  const UNPARENTHESIZED_EXPRESSION = [
    TYPE_LITERAL,
    NUMBER_LITERAL,
    BLOCK_STRING_LITERAL,
    STRING_LITERAL,
    {
      scope: 'punctuation',
      match: '(' + PUNCTUATION.map((re) => re.source).join('|') + ')',
    },
    {
      scope: 'operator',
      match: '(' + OPERATORS.map((re) => re.source).join('|') + ')',
    },
    {
      scope: 'variable.language',
      match: /\bSelf\b/,
    },
    {
      // Catch-all sub-mode at the end but excluding any nesting characters.
      match: /\w+/,
      keywords: KEYWORDS,
    },
  ];
  // Use a nesting structure so that we directly track balanced parentheses and
  // consume them. This allows patterns below to reliably "end" on closing
  // delimiters without being confused by others in the expression stream.
  // TODO: Extend this to balanced `[]`s and `{}`s.
  const PARENTHESIZED_EXPRESSION = {
    scope: 'carbon-parenthesized-expression',
    begin: /\(/,
    beginScope: 'punctuation',
    end: /\)/,
    endScope: 'punctuation',
    contains: ['self', ...UNPARENTHESIZED_EXPRESSION],
  };
  const EXPRESSION = [PARENTHESIZED_EXPRESSION, ...UNPARENTHESIZED_EXPRESSION];

  // The pattern patterns, including comma-separated sequences.
  const VALUE_PATTERN = {
    scope: 'carbon-value-pattern',
    begin: /(?!,|\)|\])/,
    end: /,|\)|\]/,
    returnEnd: true,
    contains: [...EXPRESSION],
  };
  const BINDING_PATTERN = {
    scope: 'carbon-binding-pattern',
    variants: [
      {
        begin: [/(\bvar\b)?/, /\s*/, /[a-zA-Z]\w*/, /:!?/, /\s*/],
        beginScope: {
          1: 'keyword',
          3: 'variable',
          4: 'punctuation',
        },
      },
      {
        begin: [/(\bvar\b)?/, /\s*/, /_/, /:!?/, /\s*/],
        beginScope: {
          1: 'keyword',
          3: 'keyword',
          4: 'punctuation',
        },
      },
    ],
    end: /,|\)|\]/,
    returnEnd: true,
    contains: [...EXPRESSION],
  };
  const ME_PATTERN = {
    scope: 'carbon-me-pattern',
    begin: [/(\b(addr|var)\b)?/, /\s*/, /\bme/, /:/, /\s*/],
    beginScope: {
      1: 'keyword',
      3: 'keyword',
      4: 'punctuation',
    },
    end: /,|\)|\]/,
    returnEnd: true,
    contains: [...EXPRESSION],
  };
  const UNPARENTHESIZED_PATTERNS = [ME_PATTERN, BINDING_PATTERN, VALUE_PATTERN];
  const PARENTHESIZED_PATTERN = {
    scope: 'carbon-parenthesized-pattern',
    begin: /\(/,
    beginScope: 'punctuation',
    end: /\)/,
    endScope: 'punctuation',
    contains: [
      'self',
      {
        begin: [/,/, /\s*/],
        beginScope: {
          1: 'punctuation',
        },
      },
      ...UNPARENTHESIZED_PATTERNS,
    ],
  };
  const PATTERN_SEQUENCE = [
    {
      begin: [/,/, /\s*/],
      beginScope: {
        1: 'punctuation',
      },
    },
    PARENTHESIZED_PATTERN,
    ...UNPARENTHESIZED_PATTERNS,
  ];

  // Parameters.
  const PARAMETER_LIST = {
    variants: [
      {
        scope: 'params',
        begin: /\(/,
        end: /\)/,
      },
      {
        scope: 'params',
        begin: /\[/,
        end: /\]/,
      },
    ],
    beginScope: 'punctuation',
    endScope: 'punctuation',
    contains: [...PATTERN_SEQUENCE],
  };

  // Functions.
  const FUNCTION_NAME_COMPONENT = {
    begin: [/\s*/, /\.?/, hljs.UNDERSCORE_IDENT_RE],
    beginScope: {
      2: 'punctuation',
      3: 'title.function',
    },
    end: /\s*(\.|->|;|\{)/,
    returnEnd: true,
    contains: [PARAMETER_LIST],
  };
  const RETURNED_TYPE = {
    scope: 'carbon-return-type',
    end: /\s*[;{]/,
    returnEnd: true,
    contains: [...EXPRESSION],
  };
  const RETURN_SPECIFIER = {
    begin: [/\s*/, /->/, /\s*/],
    beginScope: {
      2: 'punctuation',
    },
    starts: RETURNED_TYPE,
  };
  const FUNCTION = {
    scope: 'carbon-function',
    begin: /fn/,
    beginScope: 'keyword',
    end: /\s*[;{]/,
    returnEnd: true,
    contains: [FUNCTION_NAME_COMPONENT, RETURN_SPECIFIER],
  };

  // Classes, interfaces, and constraints.
  const CLASS_NAME_COMPONENT = {
    begin: [/\s*/, /\.?/, /(?!extends\b)[a-zA-Z_]\w*/],
    beginScope: {
      2: 'punctuation',
      3: 'title.class',
    },
    end: /\s*(\.|extends|;|\{)/,
    returnEnd: true,
    contains: [PARAMETER_LIST],
  };
  const EXTENDED_TYPE = {
    scope: 'title.class.inherited',
    end: /\s*[;{]/,
    returnEnd: true,
    contains: [...EXPRESSION],
  };
  const EXTENDS_KEYWORD = {
    begin: [/extends/, /\s+/],
    beginScope: {
      1: 'keyword',
    },
    starts: EXTENDED_TYPE,
  };
  const CLASS = {
    variants: [
      {
        scope: 'carbon-class',
        begin: [/(base|abstract)?/, /\s*/, /class/],
        beginScope: {
          1: 'keyword',
          3: 'keyword',
        },
      },
      {
        scope: 'carbon-interface',
        beginKeywords: 'interface',
      },
      {
        scope: 'carbon-constraint',
        beginKeywords: 'constraint',
      },
    ],
    end: /\s*[;{]/,
    returnEnd: true,
    contains: [CLASS_NAME_COMPONENT, EXTENDS_KEYWORD],
  };

  // Statements -- very loosely. We bundle together top-level declaration
  // constructs, blocks, and statements.
  const STATEMENTS = [
    FUNCTION,
    CLASS,
    {
      scope: 'punctuation',
      match: /;/,
    },
    ...EXPRESSION,
  ];
  const BLOCK = {
    scope: 'carbon-block',
    begin: /\{/,
    beginScope: 'punctuation',
    end: /\}/,
    endScope: 'punctuation',
    contains: ['self', ...STATEMENTS],
  };

  // Stitch everything together into the language definition.
  return {
    name: 'Carbon',
    aliases: ['carbon', 'carbon-lang'],
    keywords: KEYWORDS,
    illegal: '</',
    contains: [hljs.C_LINE_COMMENT_MODE, ...STATEMENTS, BLOCK],
  };
}
