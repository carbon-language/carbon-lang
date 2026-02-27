// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/destroy.carbon
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/ref.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/ref.carbon

// --- fail_discarded_ref.carbon

library "[[@TEST_NAME]]";

fn F() {
  var x: () = ();
  // CHECK:STDERR: fail_discarded_ref.carbon:[[@LINE+4]]:3: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   ref x;
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  ref x;
}

// --- fail_ref_out_of_place.carbon

library "[[@TEST_NAME]]";

fn F() {
  var x: () = ();
  // CHECK:STDERR: fail_ref_out_of_place.carbon:[[@LINE+4]]:22: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   let unused y: () = ref x;
  // CHECK:STDERR:                      ^~~~~
  // CHECK:STDERR:
  let unused y: () = ref x;
  // CHECK:STDERR: fail_ref_out_of_place.carbon:[[@LINE+4]]:23: error: `ref` tag is not an argument to a `ref` parameter [RefTagNoRefParam]
  // CHECK:STDERR:   let unused z: () = (ref x);
  // CHECK:STDERR:                       ^~~~~
  // CHECK:STDERR:
  let unused z: () = (ref x);
}
