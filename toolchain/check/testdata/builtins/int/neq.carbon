// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --no-dump-sem-ir
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/int/neq.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/int/neq.carbon

// --- int_neq.carbon

fn Neq(a: i32, b: i32) -> bool = "int.neq";

class True {}
class False {}

fn F(true_: True, false_: False) {
  false_ as (if Neq(1, 1) then True else False);
  true_ as (if Neq(1, 2) then True else False);
}

fn RuntimeCallIsValid(a: i32, b: i32) -> bool {
  return Neq(a, b);
}
