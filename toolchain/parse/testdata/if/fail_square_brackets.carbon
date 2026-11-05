// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/if/fail_square_brackets.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/if/fail_square_brackets.carbon

fn F() {
  // CHECK:STDERR:        ╭─┤ fail_square_brackets.carbon:[[@LINE+24]]:6
  // CHECK:STDERR: ->     │   if [] {}
  // CHECK:STDERR:        ·      ┳
  // CHECK:STDERR:        ·      ┗━━┫ error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_square_brackets.carbon:[[@LINE+18]]:5
  // CHECK:STDERR: ->     │   if [] {}
  // CHECK:STDERR:        ·     ┳
  // CHECK:STDERR:        ·     ┗━━┫ error: expected expression [ExpectedExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_square_brackets.carbon:[[@LINE+12]]:6
  // CHECK:STDERR: ->     │   if [] {}
  // CHECK:STDERR:        ·      ┳
  // CHECK:STDERR:        ·      ┗━━┫ error: expected braced code block [ExpectedCodeBlock]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_square_brackets.carbon:[[@LINE+6]]:6
  // CHECK:STDERR: ->     │   if [] {}
  // CHECK:STDERR:        ·      ┳
  // CHECK:STDERR:        ·      ┗━━┫ error: expected expression [ExpectedExpr]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  if [] {}
}

// CHECK:STDOUT: - filename: fail_square_brackets.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─FunctionIntroducer 'fn'
// CHECK:STDOUT:   │   ├─IdentifierNameMaybeBeforeSignature 'F'
// CHECK:STDOUT:   │   │ ╭─ExplicitParamListStart '('
// CHECK:STDOUT:   │   ├─ExplicitParamList ')'
// CHECK:STDOUT:   │ ╭─FunctionDefinitionStart '{'
// CHECK:STDOUT:   │ │   ╭─IfConditionStart 'if' has_error
// CHECK:STDOUT:   │ │   ├─InvalidParse '[' has_error
// CHECK:STDOUT:   │ │ ╭─IfCondition 'if' has_error
// CHECK:STDOUT:   │ │ │ ╭─CodeBlockStart '[' has_error
// CHECK:STDOUT:   │ │ │ │ ╭─InvalidParse '[' has_error
// CHECK:STDOUT:   │ │ │ ├─ExprStatement '}' has_error
// CHECK:STDOUT:   │ │ ├─CodeBlock '[' has_error
// CHECK:STDOUT:   │ ├─IfStatement 'if'
// CHECK:STDOUT:   ├─FunctionDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
