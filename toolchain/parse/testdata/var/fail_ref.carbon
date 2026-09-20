// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_ref.carbon

// CHECK:STDERR: fail_ref.carbon:[[@LINE+2]]:5-7: error: found `ref` inside `var` pattern [RefInsideVar]
// CHECK:STDERR:
var ref x: i32 = 0;

// CHECK:STDOUT: - filename: fail_ref.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameNotBeforeSignature 'x'
// CHECK:STDOUT:   │ │   ╭─RefBindingName 'ref' has_error
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':' has_error
// CHECK:STDOUT:   │ │   ├─IntTypeLiteral 'i32'
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':' has_error
// CHECK:STDOUT:   │ ├─VariablePattern 'var' has_error
// CHECK:STDOUT:   │ ├─VariableInitializer '='
// CHECK:STDOUT:   │ ├─IntLiteral '0'
// CHECK:STDOUT:   ├─VariableDecl ';'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
