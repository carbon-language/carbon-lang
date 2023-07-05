/*
 * Part of the Carbon Language project, under the Apache License v2.0 with LLVM
 * Exceptions. See /LICENSE for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

// This grammar is more permissive than toolchain because it is geared towards
// editor use.

function repeat_sep1(thing, sep) {
  return seq(thing, repeat(seq(sep, thing)));
}

function comma_sep(thing) {
  // Trailing comma is only allowed if there is atleast one element.
  return optional(seq(repeat_sep1(thing, ','), optional(',')));
}

// This is based on toolchain/parser/precedence.cpp
const PREC = {
  TermPrefix: 11,
  TermPostfix: 11,
  NumericPrefix: 10,
  NumericPostfix: 10,
  Multiplicative: 9,
  Additive: 8,
  BitwisePrefix: 7,
  BitwiseAnd: 6,
  BitwiseOr: 6,
  BitwiseXor: 6,
  BitShift: 6,
  TypePostfix: 5,
  LogicalPrefix: 4,
  Relational: 3,
  LogicalAnd: 2,
  LogicalOr: 2,
  WhereClause: 1,
  IfExpression: 1,
};

module.exports = grammar({
  name: 'carbon',

  word: ($) => $.ident,

  conflicts: ($) => [
    [$.paren_pattern, $.paren_expression],
    [$.struct_literal, $.struct_type_literal],
  ],

  extras: ($) => [/\s/, $.comment],

  // NOTE: This must match the order in src/scanner.c, names are not used for matching.
  externals: ($) => [$.binary_star, $.postfix_star],

  rules: {
    source_file: ($) =>
      seq(
        optional($.package_directive),
        repeat($.import_directive),
        repeat($.declaration)
      ),

    api_or_impl: ($) => choice('api', 'impl'),

    library_path: ($) => seq('library', $.string_literal),

    package_directive: ($) =>
      seq('package', $.ident, optional($.library_path), $.api_or_impl, ';'),

    import_directive: ($) =>
      seq('import', $.ident, optional($.library_path), ';'),

    comment: ($) => token(seq('//', /.*/)),

    ident: ($) => /[A-Za-z_][A-Za-z0-9_]*/,

    bool_literal: ($) => choice('true', 'false'),

    numeric_literal: ($) => {
      // This is using variables because rules are not allowed in token.immediate and token.
      // https://github.com/tree-sitter/tree-sitter/issues/449
      const decimal_integer_literal = choice('0', /[1-9](_?[0-9])*/);
      const hex_digits = /[0-9A-F](_?[0-9A-F])*/;
      const binary_integer_literal = /0b[01](_?[01])*/;
      const hex_integer_literal = seq('0x', token.immediate(hex_digits));

      const decimal_real_number_literal = seq(
        decimal_integer_literal,
        token.immediate(/\.[0-9](_?[0-9])*/),
        optional(
          seq(
            token.immediate(/e[+-]?/),
            token.immediate(decimal_integer_literal)
          )
        )
      );

      const hex_real_number_literal = seq(
        hex_integer_literal,
        token.immediate('.'),
        token.immediate(hex_digits),
        optional(
          seq(
            token.immediate(/p[+-]?/),
            token.immediate(decimal_integer_literal)
          )
        )
      );

      return token(
        choice(
          decimal_integer_literal,
          binary_integer_literal,
          hex_integer_literal,
          decimal_real_number_literal,
          hex_real_number_literal
        )
      );
    },

    // https://github.com/carbon-language/carbon-lang/blob/trunk/proposals/p2015.md#syntax
    numeric_type_literal: ($) => /[iuf][1-9][0-9]*/,

    _string_content: ($) => token.immediate(/[^\\"]+/),

    escape_sequence: ($) =>
      token.immediate(
        seq(
          '\\',
          choice(
            'n',
            't',
            'r',
            "'",
            '"',
            '\\',
            '0',
            /x[0-9A-F]{2}/,
            /u\{[0-9A-F]+\}/
          )
        )
      ),

    // TODO: multiline string
    string_literal: ($) =>
      seq(
        '"',
        repeat(choice($._string_content, $.escape_sequence)),
        token.immediate('"')
      ),

    array_literal: ($) =>
      seq(
        '[',
        field('type', $._expression),
        ';',
        optional(field('size', $._expression)),
        ']'
      ),

    struct_literal: ($) =>
      seq('{', comma_sep(seq($.designator, '=', $._expression)), '}'),

    struct_type_literal: ($) =>
      seq('{', comma_sep(seq($.designator, ':', $._expression)), '}'),

    builtin_type: ($) => choice('Self', 'String', 'bool', 'type'),

    literal: ($) =>
      choice(
        $.bool_literal,
        $.numeric_literal,
        $.numeric_type_literal,
        $.string_literal,
        $.struct_literal,
        $.struct_type_literal
      ),

    _binding_lhs: ($) => choice($.ident, '_'),

    paren_pattern: ($) =>
      seq('(', comma_sep($._non_expression_pattern, ','), ')'),

    _non_expression_pattern: ($) =>
      choice(
        'auto',
        seq($._binding_lhs, ':', $._expression),
        seq($._binding_lhs, ':!', $._expression),
        seq('template', $._binding_lhs, ':!', $._expression),
        $.paren_pattern,
        seq('var', $._non_expression_pattern)
      ),

    pattern: ($) => choice($._non_expression_pattern, $._expression),

    unary_prefix_expression: ($) => {
      const table = [
        [PREC.NumericPrefix, '-'],
        [PREC.NumericPrefix, '--'],
        [PREC.NumericPrefix, '++'],
        [PREC.BitwisePrefix, '^'],
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

    binary_expression: ($) => {
      const table = [
        [PREC.LogicalAnd, 'and'],
        [PREC.LogicalOr, 'or'],
        [PREC.BitwiseAnd, '&'],
        [PREC.BitwiseOr, '|'],
        [PREC.BitwiseXor, '^'],
        [PREC.BitShift, choice('<<', '>>')],
        [PREC.Relational, choice('==', '!=', '<', '<=', '>', '>=')],
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
    as_expression: ($) => prec.left(seq($._expression, 'as', $._expression)),

    ref_expression: ($) => prec.right(PREC.TermPrefix, seq('&', $._expression)),

    deref_expression: ($) =>
      prec.right(PREC.TermPrefix, seq('*', $._expression)),

    fn_type_expression: ($) =>
      prec.left(seq('__Fn', $.paren_expression, '->', $._expression)),

    if_expression: ($) =>
      prec(
        PREC.IfExpression,
        seq('if', $._expression, 'then', $._expression, 'else', $._expression)
      ),

    paren_expression: ($) => seq('(', comma_sep($._expression), ')'),

    index_expression: ($) =>
      prec(PREC.TermPostfix, seq($._expression, '[', $._expression, ']')),

    designator: ($) => seq('.', choice('base', $.ident)),

    postfix_expression: ($) =>
      prec(
        PREC.TermPostfix,
        seq(
          $._expression,
          choice(
            '++',
            '--',
            $.designator,
            seq('->', $.ident),
            seq(choice('.', '->'), '(', $._expression, ')')
          )
        )
      ),

    where_clause: ($) =>
      prec(
        PREC.WhereClause,
        choice(
          seq($._expression, '==', $._expression),
          seq($._expression, 'impls', $._expression),
          seq($.designator, '=', $._expression),
          prec.left(seq($.where_clause, 'and', $.where_clause))
        )
      ),

    where_expression: ($) =>
      prec.left(PREC.TermPostfix, seq($._expression, 'where', $.where_clause)),

    call_expression: ($) =>
      prec(PREC.TermPostfix, seq($._expression, $.paren_expression)),

    pointer_expression: ($) =>
      prec(PREC.TypePostfix, seq($._expression, $.postfix_star)),

    _expression: ($) =>
      choice(
        $.array_literal,
        $.as_expression,
        $.binary_expression,
        $.builtin_type,
        $.call_expression,
        $.deref_expression,
        $.designator,
        $.fn_type_expression,
        $.ident,
        $.if_expression,
        $.index_expression,
        $.literal,
        $.paren_expression,
        $.pointer_expression,
        $.postfix_expression,
        $.ref_expression,
        $.unary_prefix_expression,
        $.where_expression,
        seq('.', 'Self')
      ),

    var_declaration: ($) =>
      seq(
        'var',
        $._non_expression_pattern,
        optional(seq('=', $._expression)),
        ';'
      ),

    let_declaration: ($) =>
      seq('let', $._non_expression_pattern, '=', $._expression, ';'),

    assign_statement: ($) =>
      seq($._expression, $._assign_operator, $._expression, ';'),

    _assign_operator: ($) =>
      choice('=', '+=', '/=', '*=', '%=', '-=', '&=', '|=', '^=', '<<=', '>>='),

    match_clause: ($) =>
      seq(choice(seq('case', $.pattern), 'default'), '=>', $.block),

    match_statement: ($) =>
      seq('match', '(', $._expression, ')', '{', repeat($.match_clause), '}'),

    returned_var_statement: ($) => seq('returned', $.var_declaration, ';'),

    while_statement: ($) => seq('while', '(', $._expression, ')', $.block),

    break_statement: ($) => seq('break', ';'),

    continue_statement: ($) => seq('continue', ';'),

    return_statement: ($) =>
      seq('return', optional(choice('var', $._expression)), ';'),

    if_statement: ($) =>
      seq('if', '(', $._expression, ')', $.block, optional($.else)),

    else: ($) => choice(seq('else', $.if_statement), seq('else', $.block)),

    for_statement: ($) =>
      seq(
        'for',
        '(',
        $._non_expression_pattern,
        'in',
        $._expression,
        ')',
        $.block
      ),

    statement: ($) =>
      choice(
        seq($._expression, ';'),
        $.assign_statement,
        $.var_declaration,
        $.let_declaration,
        $.match_statement,
        $.returned_var_statement,
        $.if_statement,
        $.while_statement,
        $.break_statement,
        $.continue_statement,
        $.return_statement,
        $.for_statement
      ),

    _statement_list: ($) => seq($.statement, repeat($.statement)),

    block: ($) => seq('{', optional($._statement_list), '}'),

    declared_name: ($) => repeat_sep1($.ident, '.'),

    generic_binding: ($) =>
      seq(optional('template'), $.ident, ':!', $._expression),

    deduced_param: ($) =>
      choice(
        $.generic_binding,
        seq(optional('addr'), $.ident, ':', $._expression)
      ),

    deduced_params: ($) => seq('[', comma_sep($.deduced_param), ']'),

    return_type: ($) => seq('->', choice('auto', $._expression)),

    function_declaration: ($) =>
      seq(
        optional(choice('abstract', 'virtual', 'impl')),
        'fn',
        $.declared_name,
        optional($.deduced_params),
        $.paren_pattern,
        optional($.return_type),
        choice($.block, ';')
      ),

    namespace_declaration: ($) => seq('namespace', $.declared_name, ';'),

    alias_declaration: ($) =>
      seq('alias', $.declared_name, '=', $._expression, ';'),

    type_params: ($) => $.paren_pattern,

    interface_body_item: ($) =>
      choice(
        $.function_declaration,
        seq('let', $.generic_binding, ';'),
        seq('extend', $._expression, ';'),
        seq('require', $._expression, 'impls', $._expression, ';')
      ),

    interface_body: ($) => seq('{', repeat($.interface_body_item), '}'),

    interface_declaration: ($) =>
      seq(
        'interface',
        $.declared_name,
        optional($.deduced_params),
        optional($.type_params),
        $.interface_body
      ),

    constraint_declaration: ($) =>
      seq(
        'constraint',
        $.declared_name,
        optional($.deduced_params),
        optional($.type_params),
        $.interface_body
      ),

    impl_body_item: ($) => choice($.function_declaration, $.alias_declaration),

    impl_body: ($) => seq('{', $.impl_body_item, '}'),

    impl_declaration: ($) =>
      seq(
        'impl',
        optional(seq('forall', $.deduced_params)),
        'as',
        $._expression,
        $.impl_body
      ),

    extend_impl_declaration: ($) =>
      seq('extend', 'impl', 'as', $._expression, $.impl_body),

    extend_base_declaration: ($) =>
      seq('extend', 'base', ':', $._expression, ';'),

    class_body_item: ($) =>
      choice(
        $.declaration,
        $.extend_base_declaration,
        $.extend_impl_declaration
      ),

    class_body: ($) => seq('{', repeat($.class_body_item), '}'),

    class_declaration: ($) =>
      seq(
        optional(choice('base', 'abstract')),
        'class',
        $.declared_name,
        optional($.deduced_params),
        optional($.type_params),
        choice(';', $.class_body)
      ),

    empty_declaration: ($) => ';',

    declaration: ($) =>
      choice(
        $.empty_declaration,
        $.namespace_declaration,
        $.var_declaration,
        $.let_declaration,
        $.function_declaration,
        $.alias_declaration,
        $.interface_declaration,
        $.constraint_declaration,
        $.impl_declaration,
        $.class_declaration
      ),
  },
});
