// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/access/method_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/access/method_access.carbon

// --- fail_multiple_bindings.carbon

class X {
  fn F(self);
}

fn G(x: X) {
  // TODO: Produce a better diagnostic for this case.
  // CHECK:STDERR: fail_multiple_bindings.carbon:[[@LINE+2]]:3-9: error: member name of type `<bound method>` in compound member access is not an instance member or an interface member [CompoundMemberAccessDoesNotUseBase]
  // CHECK:STDERR:
  x.(x.F)();
}
