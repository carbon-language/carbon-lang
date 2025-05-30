// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=if-only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/print/int.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/print/int.carbon
// CHECK:STDERR: error: option `--dump-sem-ir-ranges=if-only` has an invalid value `if-only`; valid values are: `if-present`, `only`, or `ignore` [DriverCommandLineParseFailed]
// CHECK:STDERR:

import Core library "io";

fn Print(a: i32) = "print.int";

fn Main() {
  //@dump-sem-ir-begin
  Print(1);
  Core.Print(2);
  //@dump-sem-ir-end
}
