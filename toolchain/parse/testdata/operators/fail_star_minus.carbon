// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_star_minus.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_star_minus.carbon

// TODO: There are two possible fixes that would make this expression legal:
// `n * -n` and `n* - n`. The parser doesn't realize that the first fix is
// available because it has already accepted that first part of the expression,
// so it is recovering by using the second option, but the diagnostic should
// ideally offer (or consider) both fixes as alternatives.
// CHECK:STDERR: fail_star_minus.carbon:[[@LINE+2]]:16: error: whitespace missing after binary operator [BinaryOperatorRequiresWhitespace]
// CHECK:STDERR:
var n: i8 = n* -n;

// CHECK:STDOUT: - filename: fail_star_minus.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i8'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │   ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ │ ╭─PostfixOperatorStar '*'
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ ├─InfixOperatorMinus '-'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
