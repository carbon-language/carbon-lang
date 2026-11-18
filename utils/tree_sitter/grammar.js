/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// This grammar is more permissive than toolchain because it is geared towards
// editor use. It is based on the toolchain's parser; see in particular
// toolchain/parse/state.def and toolchain/parse/precedence.cpp.

function repeat_sep1(thing, sep) {
  return seq(thing, repeat(seq(sep, thing)));
}

function comma_sep1(thing) {
  return seq(repeat_sep1(thing, ','), optional(','));
}

function comma_sep(thing) {
  // Trailing comma is only allowed if there is at least one element.
  return optional(comma_sep1(thing));
}

// This is based on toolchain/parse/precedence.cpp. The toolchain uses a partial
// order of precedence levels, whereas tree-sitter uses a total order, so this
// is a linearization of the toolchain's order. Higher numbers bind tighter.
const PREC = {
  Postfix: 30,
  TermPrefix: 29,
  IncrementDecrement: 28,
  NumericPrefix: 27,
  BitwisePrefix: 27,
  TypePrefix: 27,
  TypePostfix: 26,
  // `like` only needs to bind less tightly than postfix `*`; it is not
  // comparable to the other operators.
  Like: 25,
  Multiplicative: 24,
  Additive: 23,
  Bitwise: 22,
  // `where` and `as` are not comparable in the toolchain. We make `where` bind
  // more tightly so that `impl T as I where ...` is not parsed as
  // `impl (T as I) where ...`.
  Requirement: 21,
  Where: 21,
  As: 20,
  Relational: 19,
  LogicalPrefix: 18,
  Logical: 17,
  Ref: 16,
  If: 15,
  Lambda: 14,
  Assignment: 13,
};

module.exports = grammar({
  name: 'carbon',

  word: ($) => $.ident,

  conflicts: ($) => [
    [$.tuple_pattern, $.paren_expression],
    [$.struct_pattern, $.struct_literal],
    [$.binding_pattern, $._primary_expression],
    [$._pattern, $.paren_expression],
    [$._pattern, $.struct_literal],
    [$.modifier, $.impl_declaration],
  ],

  extras: ($) => [/\s/, $.comment],

  // NOTE: This must match the order in src/scanner.c, names are not used for matching.
  externals: ($) => [$.binary_star, $.postfix_star, $.string],

  rules: {
    source_file: ($) => repeat($._declaration),

    comment: ($) => token(seq('//', /.*/)),

    // NOTE: this must be before ident rule to increase its priority.
    // https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p002015-numeric-type-literal-syntax.md#syntax
    numeric_type_literal: ($) => /[iuf][1-9][0-9]*/,

    ident: ($) => /(r#)?[A-Za-z_][A-Za-z0-9_]*/,

    bool_literal: ($) => choice('true', 'false'),

    // This is intentionally permissive about the digits used. A real literal
    // requires a digit after the `.`, so that `3.Foo` is not a single token.
    numeric_literal: ($) =>
      token(
        seq(
          /[0-9][0-9A-Za-z_]*/,
          optional(
            seq(/\.[0-9A-F]/, repeat(choice(/[0-9A-Za-z_]/, /[eEpP][+-]/)))
          )
        )
      ),

    char_literal: ($) =>
      token(
        seq("'", choice(/[^'\\\n]/, /\\[^\n]/, /\\u\{[0-9A-Fa-f]*\}/), "'")
      ),

    // `$0`, `$1`, ... in lambda bodies.
    positional_param: ($) => /\$[0-9]+/,

    // A tuple index in a member access, such as the `0` in `x.0`.
    tuple_index: ($) => /[0-9]+/,

    builtin_type: ($) => choice('Self', 'bool', 'char', 'str', 'type', 'auto'),

    literal: ($) =>
      choice(
        $.bool_literal,
        $.numeric_literal,
        $.char_literal,
        $.numeric_type_literal,
        $.string
      ),

    // ------------------------------------------------------------------------
    // Expressions
    // ------------------------------------------------------------------------

    designator: ($) => prec(1, seq('.', choice('base', $.ident))),

    struct_literal: ($) =>
      choice(
        seq('{', '}'),
        seq('{', comma_sep1(seq($.designator, '=', $._expression)), '}')
      ),

    struct_type_literal: ($) =>
      seq('{', comma_sep1(seq($.designator, ':', $._expression)), '}'),

    paren_expression: ($) => seq('(', comma_sep($._expression), ')'),

    array_expression: ($) =>
      seq('array', '(', $._expression, ',', $._expression, ')'),

    form_literal: ($) =>
      seq(
        'form',
        '(',
        optional(choice('val', 'var', 'ref')),
        $._expression,
        ')'
      ),

    // `.Member` or `.Self`, used in `where` clauses.
    designator_expression: ($) => seq('.', choice($.ident, 'Self')),

    // `each` is part of the name syntax, so it binds more tightly than any
    // operator. See docs/design/variadics.md.
    each_name: ($) => seq('each', $.ident),

    typeof_expression: ($) => seq('typeof', '(', $._expression, ')'),

    _primary_expression: ($) =>
      choice(
        $.ident,
        $.literal,
        $.builtin_type,
        $.positional_param,
        $.each_name,
        'self',
        'Core',
        'Cpp',
        'package',
        $.designator_expression,
        $.paren_expression,
        $.struct_literal,
        $.struct_type_literal,
        $.array_expression,
        $.form_literal,
        $.typeof_expression
      ),

    call_expression: ($) =>
      prec(
        PREC.Postfix,
        seq(
          field('function', $._expression),
          field('arguments', $.paren_expression)
        )
      ),

    member_access_expression: ($) =>
      prec(
        PREC.Postfix,
        seq(
          field('object', $._expression),
          choice('.', '->'),
          field(
            'member',
            choice($.ident, 'base', $.tuple_index, seq('(', $._expression, ')'))
          )
        )
      ),

    index_expression: ($) =>
      prec(PREC.Postfix, seq($._expression, '[', $._expression, ']')),

    pointer_type_expression: ($) =>
      prec.left(PREC.TypePostfix, seq($._expression, $.postfix_star)),

    prefix_expression: ($) => {
      const table = [
        [PREC.TermPrefix, '*'],
        [PREC.TermPrefix, '&'],
        [PREC.TermPrefix, 'expand'],
        [PREC.NumericPrefix, '-'],
        [PREC.BitwisePrefix, '^'],
        [PREC.TypePrefix, 'const'],
        [PREC.TypePrefix, 'partial'],
        [PREC.Like, 'like'],
        [PREC.LogicalPrefix, 'not'],
      ];

      return choice(
        ...table.map(([precedence, operator]) =>
          prec(
            precedence,
            seq(field('operator', operator), field('value', $._expression))
          )
        )
      );
    },

    ref_expression: ($) => prec(PREC.Ref, seq('ref', $._expression)),

    binary_expression: ($) => {
      const table = [
        [PREC.Logical, choice('and', 'or')],
        [PREC.Bitwise, choice('&', '|', '^', '<<', '>>')],
        [PREC.Relational, choice('==', '!=', '<', '<=', '>', '>=', '<=>')],
        [PREC.Additive, choice('+', '-')],
        [PREC.Multiplicative, choice($.binary_star, '/', '%')],
      ];

      return choice(
        ...table.map(([precedence, operator]) =>
          prec.left(
            precedence,
            seq(
              field('left', $._expression),
              field('operator', operator),
              field('right', $._expression)
            )
          )
        )
      );
    },

    // This should be non-associative but conflicts are not allowed in tree-sitter
    as_expression: ($) =>
      prec.left(
        PREC.As,
        seq($._expression, optional('unsafe'), 'as', $._expression)
      ),

    requirement: ($) =>
      prec.left(
        PREC.Requirement,
        seq($._expression, choice('impls', '==', '='), $._expression)
      ),

    // This is written recursively rather than with `repeat` so that the
    // precedence applies to the `and`, which would otherwise be parsed as a
    // logical `and` ending the `where` expression.
    _requirements: ($) =>
      choice(
        $.requirement,
        prec.left(PREC.Where, seq($._requirements, 'and', $._requirements))
      ),

    // Right-associative so that an `and` after a requirement continues the
    // requirement list rather than ending the `where` expression.
    where_expression: ($) =>
      prec.right(PREC.Where, seq($._expression, 'where', $._requirements)),

    if_expression: ($) =>
      prec.right(
        PREC.If,
        seq('if', $._expression, 'then', $._expression, 'else', $._expression)
      ),

    lambda_expression: ($) =>
      prec.right(
        PREC.Lambda,
        seq(
          'fn',
          optional($.implicit_parameters),
          optional($.parameters),
          optional($.return_type),
          choice(seq('=>', $._expression), $.block)
        )
      ),

    _expression: ($) =>
      choice(
        $._primary_expression,
        $.call_expression,
        $.member_access_expression,
        $.index_expression,
        $.pointer_type_expression,
        $.prefix_expression,
        $.ref_expression,
        $.binary_expression,
        $.as_expression,
        $.where_expression,
        $.if_expression,
        $.lambda_expression
      ),

    // Expressions that are only valid as complete statements.
    assignment_expression: ($) =>
      prec.right(
        PREC.Assignment,
        seq(
          $._expression,
          choice(
            '=',
            '+=',
            '-=',
            '*=',
            '/=',
            '%=',
            '&=',
            '|=',
            '^=',
            '<<=',
            '>>='
          ),
          $._expression
        )
      ),

    increment_expression: ($) =>
      prec(PREC.IncrementDecrement, seq(choice('++', '--'), $._expression)),

    // ------------------------------------------------------------------------
    // Patterns
    // ------------------------------------------------------------------------

    // In a pattern, `ref` is treated as a binding modifier rather than as the
    // start of a `ref` expression.
    binding_modifier: ($) =>
      prec(PREC.Ref + 1, choice('template', 'generic', 'runtime', 'ref')),

    binding_pattern: ($) =>
      seq(
        repeat($.binding_modifier),
        choice(
          seq(
            field('name', choice($.ident, $.each_name, 'self', '_')),
            choice(':', ':?'),
            field('type', $._expression)
          ),
          // `self` may omit its type.
          field('name', 'self')
        )
      ),

    var_pattern: ($) => seq('var', $._pattern),

    unused_pattern: ($) => seq('unused', $._pattern),

    // An element of a pattern list, with an optional default value.
    _pattern_list_element: ($) => choice($._pattern, $.default_value_pattern),

    default_value_pattern: ($) =>
      seq(
        $._pattern,
        '=',
        choice(alias('_', $.default_value_unspecified), $._expression)
      ),

    tuple_pattern: ($) => seq('(', comma_sep($._pattern_list_element), ')'),

    struct_pattern: ($) =>
      seq(
        '{',
        comma_sep(
          choice(
            seq($.designator, '=', $._pattern),
            $._pattern_list_element,
            '_'
          )
        ),
        '}'
      ),

    _pattern: ($) =>
      choice(
        $.binding_pattern,
        $.tuple_pattern,
        $.struct_pattern,
        $.var_pattern,
        $.unused_pattern,
        $._expression
      ),

    parameters: ($) => seq('(', comma_sep($._pattern_list_element), ')'),

    implicit_parameters: ($) =>
      seq('[', comma_sep($._pattern_list_element), ']'),

    // ------------------------------------------------------------------------
    // Statements
    // ------------------------------------------------------------------------

    expression_statement: ($) =>
      seq(
        choice($._expression, $.assignment_expression, $.increment_expression),
        ';'
      ),

    break_statement: ($) => seq('break', ';'),

    continue_statement: ($) => seq('continue', ';'),

    return_statement: ($) =>
      seq('return', optional(choice('var', $._expression)), ';'),

    returned_var_statement: ($) => seq('returned', $.var_declaration),

    if_statement: ($) =>
      seq(
        'if',
        '(',
        field('condition', $._expression),
        ')',
        field('then', $.block),
        optional(seq('else', field('else', choice($.if_statement, $.block))))
      ),

    while_statement: ($) =>
      seq('while', '(', field('condition', $._expression), ')', $.block),

    for_statement: ($) =>
      seq('for', '(', $._pattern, 'in', $._expression, ')', $.block),

    match_case: ($) =>
      seq(
        'case',
        $._pattern,
        optional(seq('if', field('guard', $._expression))),
        '=>',
        $.block
      ),

    match_default: ($) => seq('default', '=>', $.block),

    match_statement: ($) =>
      seq(
        'match',
        '(',
        $._expression,
        ')',
        '{',
        repeat(choice($.match_case, $.match_default)),
        '}'
      ),

    _statement: ($) =>
      choice(
        $._declaration,
        $.expression_statement,
        $.break_statement,
        $.continue_statement,
        $.return_statement,
        $.returned_var_statement,
        $.if_statement,
        $.while_statement,
        $.for_statement,
        $.match_statement
      ),

    block: ($) => seq('{', repeat($._statement), '}'),

    // ------------------------------------------------------------------------
    // Declarations
    // ------------------------------------------------------------------------

    library_specifier: ($) => seq('library', choice($.string, 'default')),

    // `impl` could be either a modifier or the start of an `impl` declaration;
    // this is handled as a conflict so that we can look further ahead, for
    // example to distinguish `impl package Foo;` from `impl package.Foo as I`.
    modifier: ($) =>
      choice(
        'abstract',
        'base',
        'default',
        'eval',
        'export',
        'extend',
        'final',
        'impl',
        'musteval',
        'override',
        'private',
        'protected',
        'static',
        'virtual',
        prec.right(seq('extern', optional($.library_specifier)))
      ),

    _modifiers: ($) => repeat1($.modifier),

    package_name: ($) => choice($.ident, 'Core', 'Cpp'),

    package_declaration: ($) =>
      seq(
        optional($._modifiers),
        'package',
        $.package_name,
        optional($.library_specifier),
        ';'
      ),

    library_declaration: ($) =>
      seq(optional($._modifiers), 'library', choice($.string, 'default'), ';'),

    import_declaration: ($) =>
      seq(
        optional($._modifiers),
        'import',
        choice(
          seq(
            $.package_name,
            optional(choice($.library_specifier, seq('inline', $.string)))
          ),
          $.library_specifier
        ),
        ';'
      ),

    name_component: ($) =>
      seq($.ident, optional($.implicit_parameters), optional($.parameters)),

    declared_name: ($) => repeat_sep1($.name_component, '.'),

    return_type: ($) => seq(choice('->', '->?'), $._expression),

    function_declaration: ($) =>
      seq(
        optional($._modifiers),
        'fn',
        $.declared_name,
        optional($.return_type),
        choice(
          ';',
          field('body', $.block),
          seq('=>', field('body', $._expression), ';'),
          seq('=', field('builtin', $.string), ';')
        )
      ),

    namespace_declaration: ($) =>
      seq(optional($._modifiers), 'namespace', $.declared_name, ';'),

    alias_declaration: ($) =>
      seq(
        optional($._modifiers),
        'alias',
        $.declared_name,
        '=',
        $._expression,
        ';'
      ),

    export_declaration: ($) =>
      seq(optional($._modifiers), 'export', $.declared_name, ';'),

    inline_declaration: ($) =>
      seq(optional($._modifiers), 'inline', 'Cpp', $.string, ';'),

    let_declaration: ($) =>
      seq(
        optional($._modifiers),
        'let',
        $._pattern,
        optional(seq('=', $._expression)),
        ';'
      ),

    var_declaration: ($) =>
      seq(
        optional($._modifiers),
        'var',
        $._pattern,
        optional(seq('=', $._expression)),
        ';'
      ),

    declaration_body: ($) => seq('{', repeat($._declaration), '}'),

    class_declaration: ($) =>
      seq(
        optional($._modifiers),
        'class',
        $.declared_name,
        choice(';', $.declaration_body)
      ),

    interface_declaration: ($) =>
      seq(
        optional($._modifiers),
        'interface',
        $.declared_name,
        choice(';', $.declaration_body)
      ),

    constraint_declaration: ($) =>
      seq(
        optional($._modifiers),
        'constraint',
        $.declared_name,
        choice(';', $.declaration_body)
      ),

    choice_alternative: ($) => seq($.ident, optional($.parameters)),

    choice_declaration: ($) =>
      seq(
        optional($._modifiers),
        'choice',
        $.declared_name,
        '{',
        comma_sep($.choice_alternative),
        '}'
      ),

    impl_declaration: ($) =>
      seq(
        optional($._modifiers),
        'impl',
        optional(seq('forall', $.implicit_parameters)),
        optional(field('type', $._expression)),
        'as',
        field('interface', $._expression),
        choice(';', $.declaration_body)
      ),

    adapt_declaration: ($) =>
      seq(optional($._modifiers), 'adapt', $._expression, ';'),

    base_declaration: ($) =>
      seq(optional($._modifiers), 'base', ':', $._expression, ';'),

    require_declaration: ($) =>
      seq(
        optional($._modifiers),
        'require',
        optional($._expression),
        'impls',
        $._expression,
        ';'
      ),

    observe_declaration: ($) =>
      seq(
        optional($._modifiers),
        'observe',
        $._expression,
        repeat(seq('impls', $._expression)),
        ';'
      ),

    friend_declaration: ($) =>
      seq(optional($._modifiers), 'friend', $._expression, ';'),

    match_first_declaration: ($) =>
      seq(optional($._modifiers), 'match_first', $.declaration_body),

    empty_declaration: ($) => ';',

    _declaration: ($) =>
      choice(
        $.empty_declaration,
        $.package_declaration,
        $.library_declaration,
        $.import_declaration,
        $.namespace_declaration,
        $.var_declaration,
        $.let_declaration,
        $.function_declaration,
        $.alias_declaration,
        $.export_declaration,
        $.inline_declaration,
        $.interface_declaration,
        $.constraint_declaration,
        $.impl_declaration,
        $.class_declaration,
        $.choice_declaration,
        $.adapt_declaration,
        $.base_declaration,
        $.require_declaration,
        $.observe_declaration,
        $.friend_declaration,
        $.match_first_declaration
      ),
  },
});
