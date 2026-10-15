// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_bad_name.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_bad_name.carbon

// CHECK:STDERR: fail_bad_name.carbon:[[@LINE+2]]:6: error: expected expression [ExpectedExpr]
// CHECK:STDERR:
var *;

// CHECK:STDERR: fail_bad_name.carbon:[[@LINE+2]]:11: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR:
var choice: i32;

// CHECK:STDERR: fail_bad_name.carbon:[[@LINE+2]]:7: error: expected name in binding pattern; prefix reserved word with `r#` to form a valid identifier [ExpectedBindingPattern]
// CHECK:STDERR:
var f2: f32;

// CHECK:STDOUT: - filename: fail_bad_name.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─InvalidParse ';' has_error
// CHECK:STDOUT:   │ │ ╭─PrefixOperatorStar '*' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'choice'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │   ╭─IdentifierNameNotBeforeSignature 'f2'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   ├─FloatTypeLiteral 'f32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
