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

// CHECK:STDERR:        ╭─┤ fail_missing_expr_after_observe.carbon:[[@LINE+6]]:9
// CHECK:STDERR: ->     │ observe impls I;
// CHECK:STDERR:        ·         ━━┳━━
// CHECK:STDERR:        ·           ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
observe impls I;

// --- fail_missing_expr_after_impls.carbon

// CHECK:STDERR:        ╭─┤ fail_missing_expr_after_impls.carbon:[[@LINE+6]]:17
// CHECK:STDERR: ->     │ observe T impls ;
// CHECK:STDERR:        ·                 ┳
// CHECK:STDERR:        ·                 ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
observe T impls ;

// --- fail_missing_impls.carbon

// CHECK:STDERR:        ╭─┤ fail_missing_impls.carbon:[[@LINE+7]]:11
// CHECK:STDERR: ->     │ observe T ;
// CHECK:STDERR:        ·           ┳
// CHECK:STDERR:        ·           ┗━━┫ error: observe should use `==` or `impls` operator
// CHECK:STDERR:        ·              ┃        [ExpectedObserveOperator]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
observe T ;

// CHECK:STDOUT: - filename: fail_missing_expr_after_observe.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ ╭─InvalidParse 'impls' has_error
// CHECK:STDOUT:   │ │ ├─IdentifierNameExpr 'I'
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls' has_error
// CHECK:STDOUT:   ├─ObserveDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_expr_after_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ │ ╭─IdentifierNameExpr 'T'
// CHECK:STDOUT:   │ │ ├─InvalidParse ';' has_error
// CHECK:STDOUT:   │ ├─ObserveImpls 'impls' has_error
// CHECK:STDOUT:   ├─ObserveDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
// CHECK:STDOUT: - filename: fail_missing_impls.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─ObserveIntroducer 'observe'
// CHECK:STDOUT:   │ ├─IdentifierNameExpr 'T'
// CHECK:STDOUT:   ├─ObserveDecl ';' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
