grammar Carbon;

input:
  package_directive import_directives declaration_list EOF
;

package_directive:
  PACKAGE IDENTIFIER optional_library_path api_or_impl SEMICOLON
;

import_directive:
  IMPORT IDENTIFIER optional_library_path SEMICOLON
;

import_directives:
  // Empty
  | import_directives import_directive
;

optional_library_path:
  // Empty
  | LIBRARY STRING_LITERAL
;

api_or_impl: API | IMPL;

primary_expression:
  IDENTIFIER
  | designator
  | PERIOD SELF
  | INTEGER_LITERAL
  | STRING_LITERAL
  | TRUE
  | FALSE
  | SIZED_TYPE_LITERAL
  | SELF
  | STRING
  | BOOL
  | TYPE
  | CONTINUATION_TYPE
  | paren_expression
  | struct_literal
  | struct_type_literal
  | LEFT_SQUARE_BRACKET expression SEMICOLON expression RIGHT_SQUARE_BRACKET
;

postfix_expression:
  primary_expression
  | postfix_expression designator
  | postfix_expression ARROW IDENTIFIER
  | postfix_expression PERIOD LEFT_PARENTHESIS expression RIGHT_PARENTHESIS
  | postfix_expression ARROW LEFT_PARENTHESIS expression RIGHT_PARENTHESIS
  | postfix_expression LEFT_SQUARE_BRACKET expression RIGHT_SQUARE_BRACKET
  | INTRINSIC_IDENTIFIER tuple
  | postfix_expression tuple
  | postfix_expression POSTFIX_STAR
  | postfix_expression UNARY_STAR
;

ref_deref_expression:
  postfix_expression
  | PREFIX_STAR ref_deref_expression
  | UNARY_STAR ref_deref_expression
  | AMPERSAND ref_deref_expression
;

fn_type_expression:
  FN_TYPE tuple ARROW type_expression
;

type_expression:
  ref_deref_expression
  | bitwise_and_expression
  | fn_type_expression
;

minus_expression:
  // ref_deref_expression excluded due to precedence diamond.
  MINUS ref_deref_expression
;

complement_expression:
  // ref_deref_expression excluded due to precedence diamond.
  CARET ref_deref_expression
;

unary_expression:
  // ref_deref_expression excluded due to precedence diamond.
  minus_expression
  | complement_expression
;

// A simple_binary_operand is an operand of a binary operator that is not itself
// a binary operator expression.
simple_binary_operand:
  ref_deref_expression
  | unary_expression
;

multiplicative_lhs:
  simple_binary_operand
  // TODO: Fix mutual lhs recursion in | multiplicative_expression
;

multiplicative_expression:
  multiplicative_lhs BINARY_STAR simple_binary_operand
  | multiplicative_lhs SLASH simple_binary_operand
;

additive_operand:
  simple_binary_operand
  | multiplicative_expression
;

additive_lhs:
  simple_binary_operand
  // TODO: Fix mutual lhs recursion in | additive_expression
;

additive_expression:
  multiplicative_expression
  | additive_lhs PLUS additive_operand
  | additive_lhs MINUS additive_operand
;

modulo_expression:
  simple_binary_operand PERCENT simple_binary_operand
;

bitwise_and_lhs:
  simple_binary_operand
  // TODO: Fix mutual lhs recursion in | bitwise_and_expression
;

bitwise_and_expression:
  bitwise_and_lhs AMPERSAND simple_binary_operand
;

bitwise_or_lhs:
  simple_binary_operand
  // TODO: Fix mutual lhs recursion in | bitwise_or_expression
;

bitwise_or_expression:
  bitwise_or_lhs PIPE simple_binary_operand
;

bitwise_xor_lhs:
  simple_binary_operand
  // TODO: Fix mutual lhs recursion in | bitwise_xor_expression
;

bitwise_xor_expression:
  bitwise_xor_lhs CARET simple_binary_operand
;

bitwise_expression:
  bitwise_and_expression
  | bitwise_or_expression
  | bitwise_xor_expression
;

bit_shift_expression:
  simple_binary_operand LESS_LESS simple_binary_operand
  | simple_binary_operand GREATER_GREATER simple_binary_operand
;

as_expression:
  simple_binary_operand AS simple_binary_operand
;

unimpl_expression:
  // ref_deref_expression excluded due to precedence diamond.
  ref_deref_expression UNIMPL_EXAMPLE ref_deref_expression
;

value_expression:
  // ref_deref_expression excluded due to precedence diamond.
  additive_expression
  | as_expression
  | bitwise_expression
  | bit_shift_expression
  | fn_type_expression
  | modulo_expression
  | unary_expression
  | unimpl_expression
;

comparison_operand:
  ref_deref_expression
  | value_expression
;

comparison_operator:
  EQUAL_EQUAL
  | LESS
  | LESS_EQUAL
  | GREATER
  | GREATER_EQUAL
  | NOT_EQUAL
;

comparison_expression:
  value_expression
  | comparison_operand comparison_operator comparison_operand
;

not_expression: NOT ref_deref_expression;

predicate_expression:
  // ref_deref_expression excluded due to precedence diamond.
  not_expression
  | comparison_expression
;

and_or_operand:
  ref_deref_expression
  | predicate_expression
;

and_lhs:
  and_or_operand
  // TODO: Fix mutual lhs recursion in | and_expression
;

and_expression:
  // predicate_expression excluded due to precedence diamond.
  and_lhs AND and_or_operand
;

or_lhs:
  and_or_operand
  // TODO: Fix mutual lhs recursion in | or_expression
;

or_expression:
  // predicate_expression excluded due to precedence diamond.
  or_lhs OR and_or_operand
;

where_clause:
  comparison_operand IS comparison_operand
  | comparison_operand EQUAL_EQUAL comparison_operand
  // TODO: .(expression) = expression
  | designator EQUAL comparison_operand
;

where_clause_list:
  where_clause
  | where_clause_list AND where_clause
;

where_expression:
  type_expression WHERE where_clause_list
;

type_or_where_expression:
  type_expression
  | where_expression
;

statement_expression:
  ref_deref_expression
  | predicate_expression
  | and_expression
  | or_expression
  | where_expression
;

if_expression:
  statement_expression
  | IF expression THEN if_expression ELSE if_expression
;

expression: if_expression;

designator: PERIOD IDENTIFIER | PERIOD BASE;

paren_expression: paren_expression_base;

tuple: paren_expression_base;

paren_expression_base:
  LEFT_PARENTHESIS RIGHT_PARENTHESIS
  | LEFT_PARENTHESIS paren_expression_contents RIGHT_PARENTHESIS
  | LEFT_PARENTHESIS paren_expression_contents COMMA RIGHT_PARENTHESIS
;

paren_expression_contents:
  expression
  | paren_expression_contents COMMA expression
;

struct_literal:
  LEFT_CURLY_BRACE RIGHT_CURLY_BRACE
  | LEFT_CURLY_BRACE struct_literal_contents RIGHT_CURLY_BRACE
  | LEFT_CURLY_BRACE struct_literal_contents COMMA RIGHT_CURLY_BRACE
;

struct_literal_contents:
  designator EQUAL expression
  | struct_literal_contents COMMA designator EQUAL expression
;

struct_type_literal:
  LEFT_CURLY_BRACE struct_type_literal_contents RIGHT_CURLY_BRACE
  | LEFT_CURLY_BRACE struct_type_literal_contents COMMA RIGHT_CURLY_BRACE
;

struct_type_literal_contents:
  designator COLON expression
  | struct_type_literal_contents COMMA designator COLON expression
;

// In many cases, using `pattern` recursively will result in ambiguities. When
// that happens, it's necessary to factor out two separate productions, one for
// when the sub-pattern is an expression, and one for when it is not. To
// facilitate this, non-terminals besides `pattern` whose names contain
// `pattern` are structured to be disjoint from `expression`, unless otherwise
// specified.
pattern: non_expression_pattern | expression;
non_expression_pattern:
  AUTO
  | binding_lhs COLON pattern
  | binding_lhs COLON_BANG expression
  | paren_pattern
  | postfix_expression tuple_pattern
  | VAR non_expression_pattern
;

binding_lhs: IDENTIFIER | UNDERSCORE;

paren_pattern: paren_pattern_base;

paren_pattern_base:
  LEFT_PARENTHESIS paren_pattern_contents RIGHT_PARENTHESIS
  | LEFT_PARENTHESIS paren_pattern_contents COMMA RIGHT_PARENTHESIS
;

// paren_pattern is analogous to paren_expression, but in order to avoid
// ambiguities, it must be disjoint from paren_expression, meaning it must
// contain at least one non_expression_pattern. The structure of this rule is
// very different from the corresponding expression rule because is has to
// enforce that requirement.
paren_pattern_contents:
  non_expression_pattern
  | paren_expression_contents COMMA non_expression_pattern
  | paren_pattern_contents COMMA expression
  | paren_pattern_contents COMMA non_expression_pattern
;

tuple_pattern: paren_pattern_base;

// Unlike most `pattern` nonterminals, this one overlaps with `expression`, so
// it should be used only when prior context (such as an introducer) rules out
// the possibility of an `expression` at this point.
maybe_empty_tuple_pattern:
  LEFT_PARENTHESIS RIGHT_PARENTHESIS
  | tuple_pattern
;

clause:
  CASE pattern DOUBLE_ARROW block
  | DEFAULT DOUBLE_ARROW block
;

clause_list:
  // Empty
  | clause_list clause
;

statement:
  assign_statement
  | VAR pattern SEMICOLON
  | VAR pattern EQUAL expression SEMICOLON
  | RETURNED VAR variable_declaration SEMICOLON
  | RETURNED VAR variable_declaration EQUAL expression SEMICOLON
  | LET pattern EQUAL expression SEMICOLON
  | statement_expression SEMICOLON
  | if_statement
  | WHILE LEFT_PARENTHESIS expression RIGHT_PARENTHESIS block
  | BREAK SEMICOLON
  | CONTINUE SEMICOLON
  | RETURN return_expression SEMICOLON
  | RETURN VAR SEMICOLON
  | MATCH LEFT_PARENTHESIS expression RIGHT_PARENTHESIS LEFT_CURLY_BRACE
    clause_list RIGHT_CURLY_BRACE
  | CONTINUATION IDENTIFIER block
  | RUN expression SEMICOLON
  | AWAIT SEMICOLON
  | FOR LEFT_PARENTHESIS variable_declaration IN type_expression
    RIGHT_PARENTHESIS block
;

assign_statement:
  statement_expression assign_operator expression SEMICOLON
  | PLUS_PLUS expression SEMICOLON
  | MINUS_MINUS expression SEMICOLON
;

assign_operator:
  EQUAL
  | PLUS_EQUAL
  | SLASH_EQUAL
  | STAR_EQUAL
  | PERCENT_EQUAL
  | MINUS_EQUAL
  | AMPERSAND_EQUAL
  | PIPE_EQUAL
  | CARET_EQUAL
  | LESS_LESS_EQUAL
  | GREATER_GREATER_EQUAL
;

if_statement:
  IF LEFT_PARENTHESIS expression RIGHT_PARENTHESIS block optional_else
;

optional_else:
  // Empty
  | ELSE if_statement
  | ELSE block
;

return_expression:
  // Empty
  | expression
;

statement_list:
  // Empty
  | statement_list statement
;

block:
  LEFT_CURLY_BRACE statement_list RIGHT_CURLY_BRACE
;

return_term:
  // Empty
  | ARROW AUTO
  | ARROW expression
;

generic_binding: IDENTIFIER COLON_BANG expression;

deduced_param:
  generic_binding
  | variable_declaration
  | ADDR variable_declaration
;

deduced_param_list:
  // Empty
  | deduced_param
  | deduced_param_list COMMA deduced_param
;

deduced_params:
  // Empty
  | LEFT_SQUARE_BRACKET deduced_param_list RIGHT_SQUARE_BRACKET
;

impl_deduced_params:
  // Empty
  | FORALL LEFT_SQUARE_BRACKET deduced_param_list RIGHT_SQUARE_BRACKET
;

// This includes the FN keyword to work around a shift-reduce conflict between virtual function's `IMPL FN` and interfaces `IMPL`.
fn_virtual_override_intro:
  FN
  | ABSTRACT FN
  | VIRTUAL FN
  | IMPL FN
;

function_declaration:
  fn_virtual_override_intro IDENTIFIER deduced_params maybe_empty_tuple_pattern
    return_term block
  | fn_virtual_override_intro IDENTIFIER deduced_params
    maybe_empty_tuple_pattern return_term SEMICOLON
;

variable_declaration: IDENTIFIER COLON pattern;

alias_declaration:
  ALIAS IDENTIFIER EQUAL expression SEMICOLON
;

// EXPERIMENTAL MIXIN FEATURE
mix_declaration: MIX expression SEMICOLON;

alternative: IDENTIFIER tuple | IDENTIFIER;

alternative_list:
  // Empty
  | alternative_list_contents
  | alternative_list_contents COMMA
;

alternative_list_contents:
  alternative
  | alternative_list_contents COMMA alternative
;

type_params:
  // Empty
  | tuple_pattern
;

// EXPERIMENTAL MIXIN FEATURE
mixin_import:
  // Empty
  | FOR expression
;

class_declaration_extensibility:
  // Empty
  | ABSTRACT
  | BASE
;

class_declaration_extends:
  // Empty
  | EXTENDS expression
;

declaration:
  NAMESPACE IDENTIFIER SEMICOLON
  | function_declaration
  | destructor_declaration
  | class_declaration_extensibility CLASS IDENTIFIER type_params
    class_declaration_extends LEFT_CURLY_BRACE class_body RIGHT_CURLY_BRACE
  | MIXIN IDENTIFIER type_params mixin_import LEFT_CURLY_BRACE mixin_body
    RIGHT_CURLY_BRACE
  | CHOICE IDENTIFIER type_params LEFT_CURLY_BRACE alternative_list
    RIGHT_CURLY_BRACE
  | VAR variable_declaration SEMICOLON
  | VAR variable_declaration EQUAL expression SEMICOLON
  | LET variable_declaration EQUAL expression SEMICOLON
  | INTERFACE IDENTIFIER type_params LEFT_CURLY_BRACE interface_body
    RIGHT_CURLY_BRACE
  | CONSTRAINT IDENTIFIER type_params LEFT_CURLY_BRACE interface_body
    RIGHT_CURLY_BRACE
  | impl_declaration
  | match_first_declaration
  | alias_declaration
;

impl_declaration:
  impl_kind_intro impl_deduced_params impl_type AS type_or_where_expression
    LEFT_CURLY_BRACE impl_body RIGHT_CURLY_BRACE
;

impl_kind_intro:
  IMPL // Internal
  | EXTERNAL IMPL
;

impl_type:
  // Self
  | type_expression
;

match_first_declaration:
  MATCH_FIRST LEFT_CURLY_BRACE match_first_declaration_list RIGHT_CURLY_BRACE
;

match_first_declaration_list:
  // Empty
  | match_first_declaration_list impl_declaration
;

destructor_declaration:
  DESTRUCTOR deduced_params block
;

declaration_list:
  // Empty
  | declaration_list declaration
;

class_body:
  // Empty
  | class_body declaration
  | class_body mix_declaration
;

// EXPERIMENTAL MIXIN FEATURE
mixin_body:
  // Empty
  | mixin_body function_declaration
  | mixin_body mix_declaration
;

interface_body:
  // Empty
  | interface_body function_declaration
  | interface_body LET generic_binding SEMICOLON
  | interface_body EXTENDS expression SEMICOLON
  | interface_body IMPL impl_type AS type_or_where_expression SEMICOLON
;

impl_body:
  // Empty
  | impl_body function_declaration
  | impl_body alias_declaration
;

ABSTRACT: 'abstract';
ADDR: 'addr';
ALIAS: 'alias';
AMPERSAND: '&';
AMPERSAND_EQUAL: '&=';
AND: 'and';
API: 'api';
ARROW: '->';
AS: 'as';
AUTO: 'auto';
AWAIT: '__await';
BASE: 'base';
BOOL: 'bool';
BREAK: 'break';
CARET: '^';
CARET_EQUAL: '^=';
CASE: 'case';
CHOICE: 'choice';
CLASS: 'class';
COLON: ':';
COLON_BANG: ':!';
COMMA: ',';
CONSTRAINT: 'constraint';
CONTINUATION: '__continuation';
CONTINUATION_TYPE: '__Continuation';
CONTINUE: 'continue';
DEFAULT: 'default';
DESTRUCTOR: 'destructor';
DOUBLE_ARROW: '=>';
ELSE: 'else';
EQUAL: '=';
EQUAL_EQUAL: '==';
EXTENDS: 'extends';
EXTERNAL: 'external';
FALSE: 'false';
FN: 'fn';
FN_TYPE: '__Fn';
FOR: 'for';
FORALL: 'forall';
GREATER: '>';
GREATER_EQUAL: '>=';
GREATER_GREATER: '>>';
GREATER_GREATER_EQUAL: '>>=';
IF: 'if';
IMPL: 'impl';
IMPORT: 'import';
IN: 'in';
INTERFACE: 'interface';
IS: 'is';
LEFT_CURLY_BRACE: '{';
LEFT_PARENTHESIS: '(';
LEFT_SQUARE_BRACKET: '[';
LESS: '<';
LESS_EQUAL: '<=';
LESS_LESS: '<<';
LESS_LESS_EQUAL: '<<=';
LET: 'let';
LIBRARY: 'library';
MATCH: 'match';
MATCH_FIRST: '__match_first';
MINUS: '-';
MINUS_EQUAL: '-=';
MINUS_MINUS: '--';
MIX: '__mix';
MIXIN: '__mixin';
NAMESPACE: 'namespace';
NOT: 'not';
NOT_EQUAL: '!=';
OR: 'or';
PACKAGE: 'package';
PERCENT: '%';
PERCENT_EQUAL: '%=';
PERIOD: '.';
PIPE: '|';
PIPE_EQUAL: '|=';
PLUS: '+';
PLUS_EQUAL: '+=';
PLUS_PLUS: '++';
RETURN: 'return';
RETURNED: 'returned';
RIGHT_CURLY_BRACE: '}';
RIGHT_PARENTHESIS: ')';
RIGHT_SQUARE_BRACKET: ']';
RUN: '__run';
SELF: 'Self';
SEMICOLON: ';';
SLASH: '/';
SLASH_EQUAL: '/=';
STAR_EQUAL: '*=';
STRING: 'String';
THEN: 'then';
TRUE: 'true';
TYPE: 'type';
UNDERSCORE: '_';
UNIMPL_EXAMPLE: '__unimplemented_example_infix';
VAR: 'var';
VIRTUAL: 'virtual';
WHERE: 'where';
WHILE: 'while';

// TODO: May need to change this.
IGNORE_WHITESPACE: [ \t\r\n]+ -> skip;

IDENTIFIER: [A-Za-z_][A-Za-z0-9_]*;
/* TODO: Remove Print special casing once we have variadics or overloads. */
INTRINSIC_IDENTIFIER: (
    'Print'
    | 'intrinsic_' [A-Za-z0-9_]*
  )
;
SIZED_TYPE_LITERAL: [iuf][1-9][0-9]*;
INTEGER_LITERAL: [0-9]+;
HORIZONTAL_WHITESPACE: [ \t\r];
WHITESPACE: [ \t\r\n];
ONE_LINE_COMMENT: '//' ~[\n]* [\n] -> skip;
OPERAND_START: [(A-Za-z0-9_"];

STRING_LITERAL: '"' [^"]+ '"';

PREFIX_STAR: ' *';
POSTFIX_STAR: '* ';
BINARY_STAR: ' * ';
UNARY_STAR: '*';
