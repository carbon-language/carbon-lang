// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_runtime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_runtime.carbon

// CHECK:STDERR: fail_runtime.carbon:[[@LINE+12]]:5: error: `runtime` is redundant here; this binding is runtime by default [RedundantRuntimeModifier]
// CHECK:STDERR: var runtime u:|
// CHECK:STDERR:     ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_runtime.carbon:[[@LINE+8]]:15: error: expected expression [ExpectedExpr]
// CHECK:STDERR: var runtime u:|
// CHECK:STDERR:               ^
// CHECK:STDERR:
// CHECK:STDERR: fail_runtime.carbon:[[@LINE+4]]:15: error: whitespace missing before binary operator [BinaryOperatorRequiresWhitespace]
// CHECK:STDERR: var runtime u:|
// CHECK:STDERR:               ^
// CHECK:STDERR:
var runtime u:|


// CHECK:STDERR: fail_runtime.carbon:[[@LINE+21]]:26: error: expected expression [ExpectedExpr]
// CHECK:STDERR: // CHECK:STDOUT:   (root)
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDERR: fail_runtime.carbon:[[@LINE+17]]:26: error: `var` declarations must end with a `;` [ExpectedDeclSemi]
// CHECK:STDERR: // CHECK:STDOUT:   (root)
// CHECK:STDERR:                          ^
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_runtime.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─VariableIntroducer 'var'
// CHECK:STDOUT:   │ │     ╭─IdentifierNameNotBeforeSignature 'u'
// CHECK:STDOUT:   │ │   ╭─RuntimeBindingName 'runtime'
// CHECK:STDOUT:   │ │   ├─BindingPatternTypeStart ':'
// CHECK:STDOUT:   │ │   │ ╭─InvalidParse '|' has_error
// CHECK:STDOUT:   │ │   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   │ │   ├─InfixOperatorPipe '|' has_error
// CHECK:STDOUT:   │ │ ╭─VarBindingPattern ':'
// CHECK:STDOUT:   │ ├─VariablePattern 'var'
// CHECK:STDOUT:   ├─VariableDecl '|' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
