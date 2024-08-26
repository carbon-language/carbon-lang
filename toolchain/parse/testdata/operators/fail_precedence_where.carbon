// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/parse/testdata/operators/fail_precedence_or_and.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/parse/testdata/operators/fail_precedence_or_and.carbon

fn F() {
  true and T where .A = B;

  U where .C impls D or false;

  e == V where .G == f32;

  W where .H = bool != i;

  X where .J < .K;

  Y as Z where .L impls M;

  S where .N == i32 as type;

  R where .O as type impls P;

  Q where .AA > .BB impls CC;
}
