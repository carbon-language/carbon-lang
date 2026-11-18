// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_variety.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_variety.carbon

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_variety.carbon:[[@LINE+42]]:21
  // CHECK:STDERR: ->     │   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: operator `=` can only be used as a complete
  // CHECK:STDERR:        ·                        ┃        statement
  // CHECK:STDERR:        ·                        ┃        [StatementOperatorAsSubExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_variety.carbon:[[@LINE+34]]:29
  // CHECK:STDERR: ->     │   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:        ·                             ┳━
  // CHECK:STDERR:        ·                             ┗━━┫ error: parentheses are required to disambiguate
  // CHECK:STDERR:        ·                                ┃        operator precedence
  // CHECK:STDERR:        ·                                ┃        [OperatorRequiresParentheses]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_variety.carbon:[[@LINE+26]]:34
  // CHECK:STDERR: ->     │   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:        ·                                  ┳
  // CHECK:STDERR:        ·                                  ┗━━┫ error: parentheses are required to
  // CHECK:STDERR:        ·                                     ┃        disambiguate operator precedence
  // CHECK:STDERR:        ·                                     ┃        [OperatorRequiresParentheses]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_variety.carbon:[[@LINE+18]]:38
  // CHECK:STDERR: ->     │   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:        ·                                      ┳
  // CHECK:STDERR:        ·                                      ┗━━┫ error: parentheses are required to
  // CHECK:STDERR:        ·                                         ┃        disambiguate operator
  // CHECK:STDERR:        ·                                         ┃        precedence
  // CHECK:STDERR:        ·                                         ┃        [OperatorRequiresParentheses]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_variety.carbon:[[@LINE+9]]:40
  // CHECK:STDERR: ->     │   n = a * b + c * d = d * d << e & f - not g;
  // CHECK:STDERR:        ·                                        ┳━━
  // CHECK:STDERR:        ·                                        ┗━━┫ error: parentheses are required
  // CHECK:STDERR:        ·                                           ┃        around this unary `not`
  // CHECK:STDERR:        ·                                           ┃        operator
  // CHECK:STDERR:        ·                                           ┃        [UnaryOperatorRequiresParentheses]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  n = a * b + c * d = d * d << e & f - not g;
}

// CHECK:STDOUT: - filename: fail_variety.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │     │   ╭─IdentifierNameExpr 'a'
// CHECK:STDOUT:   │ │     │   ├─IdentifierNameExpr 'b'
// CHECK:STDOUT:   │ │     │ ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │     │ │ ╭─IdentifierNameExpr 'c'
// CHECK:STDOUT:   │ │     │ │ ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │     │ ├─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │     ├─InfixOperatorPlus '+'
// CHECK:STDOUT:   │ │   ╭─InfixOperatorEqual '='
// CHECK:STDOUT:   │ │   │       ╭─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   │       ├─IdentifierNameExpr 'd'
// CHECK:STDOUT:   │ │   │     ╭─InfixOperatorStar '*'
// CHECK:STDOUT:   │ │   │     ├─IdentifierNameExpr 'e'
// CHECK:STDOUT:   │ │   │   ╭─InfixOperatorLessLess '<<' has_error
// CHECK:STDOUT:   │ │   │   ├─IdentifierNameExpr 'f'
// CHECK:STDOUT:   │ │   │ ╭─InfixOperatorAmp '&' has_error
// CHECK:STDOUT:   │ │   │ │ ╭─IdentifierNameExpr 'g'
// CHECK:STDOUT:   │ │   │ ├─PrefixOperatorNot 'not'
// CHECK:STDOUT:   │ │   ├─InfixOperatorMinus '-' has_error
// CHECK:STDOUT:   │ │ ╭─InfixOperatorEqual '=' has_error
// CHECK:STDOUT:   │ ├─ExprStatement ';'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
