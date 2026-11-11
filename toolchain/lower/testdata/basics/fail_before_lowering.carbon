// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/basics/fail_before_lowering.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/basics/fail_before_lowering.carbon

// This validates that earlier errors prevent lowering, without crashing.

// CHECK:STDERR:        ╭─┤ fail_before_lowering.carbon:[[@LINE+13]]:1
// CHECK:STDERR: ->     │ a;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: unrecognized declaration introducer [UnrecognizedDecl]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
// CHECK:STDERR:        ╭─┤ fail_before_lowering.carbon:[[@LINE+7]]:1
// CHECK:STDERR: ->     │ a;
// CHECK:STDERR:        · ┳
// CHECK:STDERR:        · ┗━━┫ error: semantics TODO: `handle invalid parse trees in `check``
// CHECK:STDERR:        ·    ┃        [SemanticsTodo]
// CHECK:STDERR:        ╰────
// CHECK:STDERR:
a;
