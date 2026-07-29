// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_infix_uneven_space_after.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_infix_uneven_space_after.carbon

// TODO: We could figure out that this first Failed example is infix
// with one-token lookahead.
// CHECK:STDERR: fail_infix_uneven_space_after.carbon:[[@LINE+4]]:16: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: var n: i8 = n* n;
// CHECK:STDERR:                ^
// CHECK:STDERR:
var n: i8 = n* n;

// CHECK:STDOUT: - filename: fail_infix_uneven_space_after.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'n'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i8'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'n'
// CHECK:STDOUT:   │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   ├─VariableDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
