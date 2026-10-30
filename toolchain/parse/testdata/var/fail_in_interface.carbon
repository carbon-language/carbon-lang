// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_in_interface.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_in_interface.carbon

interface I {
  // CHECK:STDERR:        ╭─┤ fail_in_interface.carbon:[[@LINE+7]]:7
  // CHECK:STDERR: ->     │   let var a: ();
  // CHECK:STDERR:        ·       ━┳━
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in associated constant declaration
  // CHECK:STDERR:        ·           ┃        [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let var a: ();

  // CHECK:STDERR:        ╭─┤ fail_in_interface.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   var b: ();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var b: ();

  // CHECK:STDERR:        ╭─┤ fail_in_interface.carbon:[[@LINE+7]]:7
  // CHECK:STDERR: ->     │   let var c: ();
  // CHECK:STDERR:        ·       ━┳━
  // CHECK:STDERR:        ·        ┗━━┫ error: expected identifier in associated constant declaration
  // CHECK:STDERR:        ·           ┃        [ExpectedAssociatedConstantIdentifier]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  let var c: ();

  // CHECK:STDERR:        ╭─┤ fail_in_interface.carbon:[[@LINE+6]]:3
  // CHECK:STDERR: ->     │   var d: ();
  // CHECK:STDERR:        ·   ━┳━
  // CHECK:STDERR:        ·    ┗━━┫ error: unrecognized declaration introducer [UnrecognizedDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var d: ();
}

// CHECK:STDOUT: - filename: fail_in_interface.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │   ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │   ├─IdentifierNameNotBeforeSignature 'I'
// CHECK:STDOUT:   │ ╭─InterfaceDefinitionStart '{'
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─InvalidParseStart 'var' has_error
// CHECK:STDOUT:   │ ├─InvalidParseSubtree ';' has_error
// CHECK:STDOUT:   │ │ ╭─AssociatedConstantIntroducer 'let'
// CHECK:STDOUT:   │ ├─AssociatedConstantDecl ';' has_error
// CHECK:STDOUT:   │ │ ╭─InvalidParseStart 'var' has_error
// CHECK:STDOUT:   │ ├─InvalidParseSubtree ';' has_error
// CHECK:STDOUT:   ├─InterfaceDefinition '}'
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
