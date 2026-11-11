// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/var/fail_runtime.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/var/fail_runtime.carbon

// CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+33]]:5
// CHECK:STDERR: ->     │ var runtime u:|
// CHECK:STDERR:        ·     ━┳━━━━━
// CHECK:STDERR:        ·      ┗━━┫ error: `runtime` is redundant here; this binding is runtime by default
// CHECK:STDERR:        ·         ┃        [RedundantRuntimeModifier]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+26]]:15
// CHECK:STDERR: ->     │ var runtime u:|
// CHECK:STDERR:        ·               ┳
// CHECK:STDERR:        ·               ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+20]]:15
// CHECK:STDERR: ->     │ var runtime u:|
// CHECK:STDERR:        ·               ┳
// CHECK:STDERR:        ·               ┗━━┫ error: whitespace missing before binary operator
// CHECK:STDERR:        ·                  ┃        [BinaryOperatorRequiresWhitespace]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+13]]:16
// CHECK:STDERR: ->     │ var runtime u:|
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: expected expression [ExpectedExpr]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_runtime.carbon:[[@LINE+7]]:16
// CHECK:STDERR: ->     │ var runtime u:|
// CHECK:STDERR:        ·                ┳
// CHECK:STDERR:        ·                ┗━━┫ error: `var` declarations must end with a `;`
// CHECK:STDERR:        ·                   ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
var runtime u:|


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
