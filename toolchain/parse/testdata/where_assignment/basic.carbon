// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/where_assignment/basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/where_assignment/basic.carbon

impl C as Z {
  // CHECK:STDERR:        ╭─┤ basic.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   where Z1 = D.(Y.Y1);
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z1 = D.(Y.Y1);
  // CHECK:STDERR:        ╭─┤ basic.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   where Z2 = Z1;
  // CHECK:STDERR:        ·   ━━┳━━
  // CHECK:STDERR:        ·     ┗━━┫ error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  where Z2 = Z1;
}

// CHECK:STDOUT: - filename: basic.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─ImplIntroducer 'impl'
// CHECK:STDOUT:   │   │ ╭─IdentifierNameExpr 'C'
// CHECK:STDOUT:   │   ├─ImplTypeAs 'as'
// CHECK:STDOUT:   │   ├─IdentifierNameExpr 'Z'
// CHECK:STDOUT:   │ ╭─ImplDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─InvalidParseStart 'where' has_error
// CHECK:STDOUT:   │ ├─InvalidParseSubtree ';' has_error
// CHECK:STDOUT:   │ │ ╭─InvalidParseStart 'where' has_error
// CHECK:STDOUT:   │ ├─InvalidParseSubtree ';' has_error
// CHECK:STDOUT:   ├─ImplDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
