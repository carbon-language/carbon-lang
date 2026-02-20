// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/symbolic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/symbolic.carbon

// --- symbolic_call_to_eval_fn.carbon

library "[[@TEST_NAME]]";

eval fn F(B:! bool) -> type {
  if (B) { return (); }
  return ({}, {});
}

fn G(B:! bool) {
  let unused n: F(B) = ({}, {});
}

fn H() {
  G(false);
}


// --- fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon
// TODO: Provide a location for the error message.

library "[[@TEST_NAME]]";

eval fn F(B:! bool) -> type {
  if (B) { return (); }
  return ({}, {});
}

fn G(B:! bool) {
  let unused n: F(B) = ({}, {});
}

fn H() {
  // CHECK:STDERR: fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon:[[@LINE+5]]:3: error: unable to monomorphize specific `G(true)` [ResolvingSpecificHere]
  // CHECK:STDERR:   G(true);
  // CHECK:STDERR:   ^
  // CHECK:STDERR: fail_symbolic_call_to_eval_fn_invalid_instantiation.carbon: note: cannot initialize tuple of 0 elements from tuple with 2 elements [TupleInitElementCountMismatch]
  // CHECK:STDERR:
  G(true);
}
