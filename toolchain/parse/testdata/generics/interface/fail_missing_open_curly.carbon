// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/generics/interface/fail_missing_open_curly.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/generics/interface/fail_missing_open_curly.carbon

// CHECK:STDERR:        ╭─┤ fail_missing_open_curly.carbon:[[@LINE+8]]:15
// CHECK:STDERR: ->     │ interface Bar Baz {}
// CHECK:STDERR:        ·               ┳━━
// CHECK:STDERR:        ·               ┗━━┫ error: `interface` declarations must either end with a `;` or
// CHECK:STDERR:        ·                  ┃        have a `{ ... }` block for a definition
// CHECK:STDERR:        ·                  ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
interface Bar Baz {}

interface Foo

// CHECK:STDERR:        ╭─┤ fail_missing_open_curly.carbon:[[@LINE+18]]:26
// CHECK:STDERR: ->     │ // CHECK:STDOUT:   (root)
// CHECK:STDERR:        ·                          ┳
// CHECK:STDERR:        ·                          ┗━━┫ error: `interface` declarations must either end
// CHECK:STDERR:        ·                             ┃        with a `;` or have a `{ ... }` block for a
// CHECK:STDERR:        ·                             ┃        definition
// CHECK:STDERR:        ·                             ┃        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_missing_open_curly.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Bar'
// CHECK:STDOUT:   ├─InterfaceDecl '}' has_error
// CHECK:STDOUT:   │ ╭─InterfaceIntroducer 'interface'
// CHECK:STDOUT:   │ ├─IdentifierNameNotBeforeSignature 'Foo'
// CHECK:STDOUT:   ├─InterfaceDecl 'Foo' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
