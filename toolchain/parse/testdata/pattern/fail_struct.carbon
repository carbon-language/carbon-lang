// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/pattern/fail_struct.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/pattern/fail_struct.carbon

fn F() {
  // Shorthand expected to fail with a TODO
  let {a: i32} = {.a = 1};

  // Expected `.field = pattern` in struct pattern
  let {1} = {.a = 1};

  // Expected `=` after field designator in struct pattern
  let {.a 1} = {.a = 1};

  // Recovery test (skips to next `,` or `}`)
  let {.a 1, .b = b: i32} = {.a = 1, .b = 2};
}
