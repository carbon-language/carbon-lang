// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/let/fail_no_semi.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/let/fail_no_semi.carbon

let

// CHECK:STDERR:        ╭─┤ fail_no_semi.carbon:[[@LINE+19]]:26
// CHECK:STDERR: ->     │ // CHECK:STDOUT:   (root)
// CHECK:STDERR:        ·                          ┳
// CHECK:STDERR:        ·                          ┗━━┫ error: expected pattern [ExpectedPattern]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_no_semi.carbon:[[@LINE+13]]:26
// CHECK:STDERR: ->     │ // CHECK:STDOUT:   (root)
// CHECK:STDERR:        ·                          ┳
// CHECK:STDERR:        ·                          ┗━━┫ error: `let` declarations must end with a `;`
// CHECK:STDERR:        ·                             ┃        [ExpectedDeclSemi]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDOUT: - filename: fail_no_semi.carbon
// CHECK:STDOUT:   ╭─FileStart ''
// CHECK:STDOUT:   │ ╭─LetIntroducer 'let'
// CHECK:STDOUT:   │ ├─InvalidParse '' has_error
// CHECK:STDOUT:   ├─LetDecl 'let' has_error
// CHECK:STDOUT:   ├─FileEnd ''
// CHECK:STDOUT:   (root)
