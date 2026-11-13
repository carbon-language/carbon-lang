// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/return/fail_no_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/return/fail_no_semi.carbon

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_no_semi.carbon:[[@LINE+13]]:9
  // CHECK:STDERR: ->     │   return
  // CHECK:STDERR:        ·         ┳
  // CHECK:STDERR:        ·         ┗━━┫ error: expected expression [ExpectedExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_no_semi.carbon:[[@LINE+7]]:9
  // CHECK:STDERR: ->     │   return
  // CHECK:STDERR:        ·         ┳
  // CHECK:STDERR:        ·         ┗━━┫ error: `return` statements must end with a `;`
  // CHECK:STDERR:        ·            ┃        [ExpectedStatementSemi]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  return
}

// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─ReturnStatementStart 'return'
// CHECK:STDOUT:   │ │ ├─InvalidParse '}' has_error
// CHECK:STDOUT:   │ ├─ReturnStatement 'return' has_error
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
