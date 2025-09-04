// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/type/destroy_struct_and_tuple.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/type/destroy_struct_and_tuple.carbon

// --- forall.carbon
library "[[@TEST_NAME]]";

interface DestroyLike {
  fn Op[addr self: Self*]();
}

impl forall [T:! type] T as DestroyLike {
  fn Op[addr self: Self*]() = "type.destroy_struct_and_tuple";
}

var a: ();
var b: {};

fn F() {
  a.(DestroyLike.Op)();
  b.(DestroyLike.Op)();
}

// --- fail_wrong_type.carbon
library "[[@TEST_NAME]]";

interface DestroyLike {
  fn Op[addr self: Self*]();
}

class C {}

impl C as DestroyLike {
  // CHECK:STDERR: fail_wrong_type.carbon:[[@LINE+4]]:3: error: invalid signature for builtin function "type.destroy_struct_and_tuple" [InvalidBuiltinSignature]
  // CHECK:STDERR:   fn Op[addr self: Self*]() = "type.destroy_struct_and_tuple";
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  fn Op[addr self: Self*]() = "type.destroy_struct_and_tuple";
}
