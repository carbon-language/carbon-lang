// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function repeat_sep1(thing, sep) {
  return seq(thing, repeat(seq(sep, thing)));
}

function repeat_sep(thing, sep) {
  return optional(seq(thing, repeat(seq(sep, thing))));
}

// TODO: very arbitrary, need to be correct
const PREC = {
  paren: 17,
  if_expr: 16,
  where: 15,
  call: 14,
  postfix: 13,
  unary: 12,
  cast: 11,
  multiplicative: 10,
  additive: 9,
  shift: 8,
  bitand: 7,
  bitxor: 6,
  bitor: 5,
  comparative: 4,
  and: 3,
  or: 2,
  assign: 0,
};

module.exports = grammar({
  name: 'carbon',

  conflicts: ($) => [
    [$.paren_pattern, $.paren_expression],
    [$.struct_literal, $.struct_type_literal],
  ],
  extras: ($) => [/\s/, $.comment],

  // NOTE: must match the order in src/scanner.c, the names don't matter
  externals: ($) => [
    $.binary_star,
    $.prefix_star,
    $.postfix_star,
    $.unary_star,
  ],

  rules: {
    source_file: ($) =>
      seq(
        optional($.package_directive),
        repeat($.import_directive),
        repeat($.declaration)
      ),
    comment: ($) => token(seq('//', /.*/)),
    ident: ($) => /[A-Za-z_][A-Za-z0-9_]*/,
    bool_literal: ($) => choice('true', 'false'),
    integer_literal: ($) => /[0-9]+/,
    float_literal: ($) => /[0-9\.]+/,
    sized_type_literal: ($) => /[iuf][1-9][0-9]*/,
    string_literal: ($) => /"[^"]*"/, // TODO
    array_literal: ($) =>
      seq(
        '[',
        field('type', $._expression),
        ';',
        optional(field('size', $._expression)),
        ']'
      ),
    struct_literal: ($) =>
      seq(
        '{',
        repeat_sep(seq($.designator, '=', $._expression), ','),
        optional(','),
        '}'
      ),
    struct_type_literal: ($) =>
      seq(
        '{',
        repeat_sep(seq($.designator, ':', $._expression), ','),
        optional(','),
        '}'
      ),

    builtin_type: ($) => choice('Self', 'String', 'bool', 'type', 'i32'),
    literal: ($) =>
      choice(
        $.bool_literal,
        $.integer_literal,
        $.string_literal,
        $.sized_type_literal,
        $.float_literal,
        $.struct_literal,
        $.struct_type_literal
      ),

    _binding_lhs: ($) => seq(optional('addr'), choice($.ident, '_')),
    paren_pattern: ($) =>
      seq('(', repeat_sep($._non_expression_pattern, ','), optional(','), ')'),
    _non_expression_pattern: ($) =>
      choice(
        'auto',
        seq($._binding_lhs, ':', $._expression),
        seq($._binding_lhs, ':!', $._expression),
        seq('template', $._binding_lhs, ':!', $._expression),
        $.paren_pattern,
        // TODO: where is this used
        // seq(
        //   $.postfix_expression, $.tuple_pattern,
        // ),
        seq('var', $._non_expression_pattern)
      ),
    pattern: ($) => choice($._non_expression_pattern, $._expression),

    unary_expression: ($) =>
      prec(PREC.unary, seq(choice('-', 'not', '^'), $._expression)),

    binary_expression: ($) => {
      const table = [
        [PREC.and, 'and'],
        [PREC.or, 'or'],
        [PREC.bitand, '&'],
        [PREC.bitor, '|'],
        [PREC.bitxor, '^'],
        [PREC.comparative, choice('==', '!=', '<', '<=', '>', '>=')],
        [PREC.shift, choice('<<', '>>')],
        [PREC.additive, choice('+', '-')],
        [PREC.multiplicative, choice($.binary_star, '/', '%')],
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

    as_expression: ($) =>
      prec.left(PREC.cast, seq($._expression, 'as', $._expression)),

    ref_expression: ($) => prec.right(PREC.unary, seq('&', $._expression)),
    deref_expression: ($) =>
      prec.right(PREC.unary, seq($.prefix_star, $._expression)),
    fn_type_expression: ($) =>
      prec.left(seq('fn', $.tuple, '->', $._expression)),

    if_expression: ($) =>
      prec(
        PREC.if_expr,
        seq('if', $._expression, 'then', $._expression, 'else', $._expression)
      ),

    paren_expression: ($) =>
      seq('(', repeat_sep($._expression, ','), optional(','), ')'),
    tuple: ($) => $.paren_expression,

    index_expression: ($) =>
      prec(PREC.postfix, seq($._expression, '[', $._expression, ']')),
    designator: ($) => seq('.', choice('base', $.ident)),
    postfix_expression: ($) =>
      prec(
        PREC.postfix,
        seq(
          $._expression,
          choice(
            $.designator,
            seq('->', $.ident),
            seq(choice('.', '->'), '(', $._expression, ')')
          )
        )
      ),
    where_clause: ($) =>
      choice(
        seq($._expression, '==', $._expression),
        seq($._expression, 'impls', $._expression),
        seq($.designator, '=', $._expression),
        prec.left(seq($.where_clause, choice('or', 'and'), $.where_clause))
      ),
    where_expression: ($) =>
      prec(PREC.where, seq($._expression, 'where', $.where_clause)),
    call_expression: ($) => prec(PREC.call, seq($._expression, $.tuple)),
    pointer_expression: ($) =>
      prec(PREC.postfix, seq($._expression, $.postfix_star)),
    _expression: ($) =>
      choice(
        $.ident,
        $.pointer_expression,
        $.index_expression,
        $.fn_type_expression,
        $.designator,
        seq('.', 'Self'),
        $.literal,
        $.builtin_type,
        $.paren_expression,
        $.array_literal,
        $.if_expression,
        $.binary_expression,
        $.unary_expression,
        $.as_expression,
        $.paren_expression,
        $.postfix_expression,
        $.call_expression,
        $.ref_expression,
        $.deref_expression,
        $.where_expression
      ),

    _variable_declaration_inner: ($) => seq($.ident, ':', $._expression),
    variable_declaration: ($) =>
      choice(
        seq('var', $.pattern, optional(seq('=', $._expression)), ';'),
        seq('let', $.pattern, '=', $._expression, ';')
      ),

    clause: ($) =>
      seq(choice(seq('case', $.pattern), 'default'), '=>', $.block),
    clause_list: ($) => seq($.clause_list, $.clause),

    assign_statement: ($) =>
      choice(
        seq($._expression, $._assign_operator, $._expression, ';'),
        seq('++', $._expression, ';'),
        seq('--', $._expression, ';')
      ),

    _assign_operator: ($) =>
      choice('=', '+=', '/=', '*=', '%=', '-=', '&=', '|=', '^=', '<<=', '>>='),

    match_statement: ($) =>
      seq('match', '(', $._expression, ')', '{', $.clause_list, '}'),
    returned_var_statement: ($) =>
      seq(
        'returned',
        'var',
        $._variable_declaration_inner,
        optional(seq('=', $._expression)),
        ';'
      ),
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
        optional(choice('var', 'let')), // not in explorer
        $._variable_declaration_inner,
        'in',
        $._expression,
        ')',
        $.block
      ),
    statement: ($) =>
      choice(
        seq($._expression, ';'),
        $.assign_statement,
        $.variable_declaration,
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
    deduced_params: ($) =>
      seq('[', repeat_sep($.deduced_param, ','), optional(','), ']'),

    tuple_pattern: ($) => $.paren_pattern,
    return_type: ($) => seq('->', choice('auto', $._expression)),
    function_declaration: ($) =>
      seq(
        optional(choice('abstract', 'virtual', 'impl')),
        'fn',
        $.declared_name,
        optional($.deduced_params),
        $.tuple_pattern,
        optional($.return_type),
        choice($.block, ';')
      ),
    namespace_declaration: ($) => seq('namespace', $.declared_name, ';'),
    alias_declaration: ($) =>
      seq('alias', $.declared_name, '=', $._expression, ';'),
    type_params: ($) => $.tuple_pattern,
    interface_body_item: ($) =>
      choice(
        $.function_declaration,
        seq('let', $.generic_binding, ';'),
        seq('extends', $._expression, ';'),
        seq('impl', $._expression, 'as', $._expression, ';')
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
    class_body_item: ($) => $.declaration,
    class_body: ($) => seq('{', repeat($.class_body_item), '}'),
    class_declaration: ($) =>
      seq(
        optional(choice('base', 'abstract')),
        'class',
        $.declared_name,
        optional($.deduced_params),
        optional($.type_params),
        optional(seq('extends', $._expression)),
        choice(';', $.class_body)
      ),
    declaration: ($) =>
      choice(
        $.namespace_declaration,
        $.variable_declaration,
        $.function_declaration,
        $.alias_declaration,
        $.interface_declaration,
        $.constraint_declaration,
        $.class_declaration,
        ';' // empty declaration
      ),

    api_or_impl: ($) => choice('api', 'impl'),
    library_path: ($) => seq('library', $.string_literal),
    package_directive: ($) =>
      seq('package', $.ident, optional($.library_path), $.api_or_impl, ';'),
    import_directive: ($) =>
      seq('import', $.ident, optional($.library_path), ';'),
  },
});
