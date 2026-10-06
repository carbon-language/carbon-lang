// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_star_star_no_space.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_star_star_no_space.carbon

// TODO: We try to form (n*)*p and reject due to missing parentheses
// before we notice the missing whitespace around the second `*`.
// It'd be better to (somehow) form n*(*p) and reject due to the missing
// whitespace around the first `*`.
// CHECK:STDERR: fail_star_star_no_space.carbon:[[@LINE+4]]:16: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: var n: i8 = n**p;
// CHECK:STDERR:                ^
// CHECK:STDERR:
var n: i8 = n**p;

// CHECK:STDOUT: - filename: fail_star_star_no_space.carbon
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
// CHECK:STDOUT:   │ ├─PostfixOperatorStar '*'
// CHECK:STDOUT:   ├─VariableDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
