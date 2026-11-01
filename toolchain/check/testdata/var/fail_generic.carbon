// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_generic.carbon

fn Main() {
  // CHECK:STDERR:        ╭─┤ fail_generic.carbon:[[@LINE+15]]:15
  // CHECK:STDERR: ->     │   var generic x: () = ();
  // CHECK:STDERR:        ·               ┳━━━━
  // CHECK:STDERR:        ·               ┗━━┫ error: semantics TODO:
  // CHECK:STDERR:        ·                  ┃        `handle invalid parse trees in `check``
  // CHECK:STDERR:        ·                  ┃        [SemanticsTodo]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  // CHECK:STDERR:        ╭─┤ fail_generic.carbon:[[@LINE+7]]:21
  // CHECK:STDERR: ->     │   var generic x: () = ();
  // CHECK:STDERR:        ·                     ┳
  // CHECK:STDERR:        ·                     ┗━━┫ error: found generic binding inside `var` pattern
  // CHECK:STDERR:        ·                        ┃        [NonRegularBindingInVarDecl]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var generic x: () = ();
}

// CHECK:STDOUT: --- fail_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main();
// CHECK:STDOUT:
