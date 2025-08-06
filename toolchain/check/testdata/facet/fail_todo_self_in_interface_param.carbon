// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/fail_todo_self_in_interface_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/fail_todo_self_in_interface_param.carbon

interface I(T:! type) {
  let I1:! type;
}

// @dump-sem-ir-begin
// CHECK:STDERR: fail_todo_self_in_interface_param.carbon:[[@LINE+4]]:12: error: name `.Self` not found [NameNotFound]
// CHECK:STDERR: fn F(T:! I(.Self) where .I1 = ()) -> T.I1 {
// CHECK:STDERR:            ^~~~~
// CHECK:STDERR:
fn F(T:! I(.Self) where .I1 = ()) -> T.I1 {
  return ();
}
// @dump-sem-ir-end
