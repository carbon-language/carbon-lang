// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/inheritance/fail_init_missing_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/inheritance/fail_init_missing_base.carbon

base class X {}

class Z {
  extend base: X;
}

fn Run() {
  // CHECK:STDERR:        ╭─┤ fail_init_missing_base.carbon:[[@LINE+7]]:14
  // CHECK:STDERR: ->     │   var _: Z = {};
  // CHECK:STDERR:        ·              ┳━
  // CHECK:STDERR:        ·              ┗━━┫ error: missing value for field `base` in struct initialization
  // CHECK:STDERR:        ·                 ┃        [StructInitMissingFieldInLiteral]
  // CHECK:STDERR:        ╰────
  // CHECK:STDERR:
  var _: Z = {};
}
