// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/array/fail_syntax.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/array/fail_syntax.carbon

// --- fail_no_comma.carbon

// CHECK:STDERR:        ╭─┤ fail_no_comma.carbon:[[@LINE+13]]:17
// CHECK:STDERR: ->     │ var x: array(i32);
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected `,` in `array(Type, Count)`
// CHECK:STDERR:        ·                    ┃        [ExpectedArrayComma]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_no_comma.carbon:[[@LINE+6]]:17
// CHECK:STDERR: ->     │ var x: array(i32);
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array(i32);

// --- fail_invalid_char.carbon

// CHECK:STDERR:        ╭─┤ fail_invalid_char.carbon:[[@LINE+26]]:17
// CHECK:STDERR: ->     │ fn X() -> array(:);
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_invalid_char.carbon:[[@LINE+20]]:17
// CHECK:STDERR: ->     │ fn X() -> array(:);
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected `,` in `array(Type, Count)`
// CHECK:STDERR:        ·                    ┃        [ExpectedArrayComma]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_invalid_char.carbon:[[@LINE+13]]:17
// CHECK:STDERR: ->     │ fn X() -> array(:);
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_invalid_char.carbon:[[@LINE+7]]:17
// CHECK:STDERR: ->     │ fn X() -> array(:);
// CHECK:STDERR:        ·                ╷╻
// CHECK:STDERR:        ·                │┗━━┫ error: unexpected tokens before `)` [ExpectedCloseSymbol]
// CHECK:STDERR:        ·                ╰─────┤ matching this `(` [GroupOpenedHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
fn X() -> array(:);

// --- fail_unlexed_expr.carbon

// CHECK:STDERR:        ╭─┤ fail_unlexed_expr.carbon:[[@LINE+21]]:14
// CHECK:STDERR: ->     │ var y: array(`);
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: encountered unrecognized characters while parsing
// CHECK:STDERR:        ·                 ┃        [UnrecognizedCharacters]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_unlexed_expr.carbon:[[@LINE+14]]:14
// CHECK:STDERR: ->     │ var y: array(`);
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected `,` in `array(Type, Count)`
// CHECK:STDERR:        ·                 ┃        [ExpectedArrayComma]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_unlexed_expr.carbon:[[@LINE+7]]:14
// CHECK:STDERR: ->     │ var y: array(`);
// CHECK:STDERR:        ·             ╷╻
// CHECK:STDERR:        ·             │┗━━┫ error: unexpected tokens before `)` [ExpectedCloseSymbol]
// CHECK:STDERR:        ·             ╰─────┤ matching this `(` [GroupOpenedHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var y: array(`);

// --- fail_no_open_paren.carbon

// CHECK:STDERR:        ╭─┤ fail_no_open_paren.carbon:[[@LINE+6]]:14
// CHECK:STDERR: ->     │ var x: array i32, 1;
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected `(` after `array` [ExpectedParenAfter]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array i32, 1;

// --- fail_no_close_paren.carbon

// CHECK:STDERR:        ╭─┤ fail_no_close_paren.carbon:[[@LINE+15]]:13
// CHECK:STDERR: ->     │ var x: array(i32,;
// CHECK:STDERR:        ·             ┳    ┬
// CHECK:STDERR:        ·             ┗━━━━━━━┫ error: opening symbol without a corresponding closing
// CHECK:STDERR:        ·                  │  ┃        symbol
// CHECK:STDERR:        ·                  │  ┃        [UnmatchedOpening]
// CHECK:STDERR:        ·                  ╰────┤ possibly missing `)` here [PossiblyMissingBracketHere]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_no_close_paren.carbon:[[@LINE+6]]:18
// CHECK:STDERR: ->     │ var x: array(i32,;
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array(i32,;

// --- fail_no_length.carbon

// CHECK:STDERR:        ╭─┤ fail_no_length.carbon:[[@LINE+6]]:18
// CHECK:STDERR: ->     │ var x: array(i32,);
// CHECK:STDERR:        ·                  ┳
// CHECK:STDERR:        ·                  ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array(i32,);

// --- fail_no_params.carbon

// CHECK:STDERR:        ╭─┤ fail_no_params.carbon:[[@LINE+19]]:14
// CHECK:STDERR: ->     │ var x: array();
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_no_params.carbon:[[@LINE+13]]:14
// CHECK:STDERR: ->     │ var x: array();
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected `,` in `array(Type, Count)`
// CHECK:STDERR:        ·                 ┃        [ExpectedArrayComma]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_no_params.carbon:[[@LINE+6]]:14
// CHECK:STDERR: ->     │ var x: array();
// CHECK:STDERR:        ·              ┳
// CHECK:STDERR:        ·              ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array();

// --- fail_keyword_alone.carbon

// CHECK:STDERR:        ╭─┤ fail_keyword_alone.carbon:[[@LINE+25]]:13
// CHECK:STDERR: ->     │ var x: array;
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: expected `(` after `array` [ExpectedParenAfter]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_keyword_alone.carbon:[[@LINE+19]]:13
// CHECK:STDERR: ->     │ var x: array;
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_keyword_alone.carbon:[[@LINE+13]]:13
// CHECK:STDERR: ->     │ var x: array;
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: expected `,` in `array(Type, Count)`
// CHECK:STDERR:        ·                ┃        [ExpectedArrayComma]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_keyword_alone.carbon:[[@LINE+6]]:13
// CHECK:STDERR: ->     │ var x: array;
// CHECK:STDERR:        ·             ┳
// CHECK:STDERR:        ·             ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var x: array;

// CHECK:STDOUT: - filename: fail_no_comma.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ')' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_invalid_char.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │ ├─IdentifierNameMaybeBeforeSignature 'X'
// CHECK:STDOUT:   │ │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │ ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ │   ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   ├─InvalidParse ':' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExprComma ':' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse ':' has_error
// CHECK:STDOUT:   │ │ ╭─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ ├─ReturnType '->'
// CHECK:STDOUT:   ├─FunctionDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_unlexed_expr.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'y'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '`' has_error
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma '`' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '`' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_open_paren.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen 'array' has_error
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │   │ ├─IntLiteral '1'
// CHECK:STDOUT:   │ │   ├─ArrayExpr 'array' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_close_paren.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_length.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ','
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_no_params.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen '('
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ')' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ')' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr ')' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_keyword_alone.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─ArrayExprKeyword 'array'
// CHECK:STDOUT:   │ │   │ ├─ArrayExprOpenParen 'array' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │   │ ├─ArrayExprComma ';' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │   ├─ArrayExpr 'array' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
