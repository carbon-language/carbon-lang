// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/observe/fail_missing_expr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/observe/fail_missing_expr.carbon

// --- fail_missing_expr_after_observe.carbon

// CHECK:STDERR: fail_missing_expr_after_observe.carbon:[[@LINE+4]]:9: error: expected expression [ExpectedExpr]
// CHECK:STDERR: observe impls I;
// CHECK:STDERR:         ^~~~~
// CHECK:STDERR:
observe impls I;

// --- fail_missing_expr_after_impls.carbon

// CHECK:STDERR: fail_missing_expr_after_impls.carbon:[[@LINE+4]]:17: error: expected expression [ExpectedExpr]
// CHECK:STDERR: observe T impls ;
// CHECK:STDERR:                 ^
// CHECK:STDERR:
observe T impls ;

// --- fail_missing_impls.carbon

// CHECK:STDERR: fail_missing_impls.carbon:[[@LINE+4]]:11: error: observe should use `==` or `impls` operator [ExpectedObserveOperator]
// CHECK:STDERR: observe T ;
// CHECK:STDERR:           ^
// CHECK:STDERR:
observe T ;

// CHECK:STDOUT: - filename: fail_missing_expr_after_observe.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ ╭─InvalidParse 'impls' ❌
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls' ❌
// CHECK:STDOUT:   ├─ObserveDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_expr_after_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' ❌
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls' ❌
// CHECK:STDOUT:   ├─ObserveDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   ├─ObserveDecl ';' ❌
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
