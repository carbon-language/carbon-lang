// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/overloaded/fail_no_impl_for_arg.carbon

package User;

class C {};
class D {};

impl forall [T:! Core.ImplicitAs(C)] C as Core.AddWith(T) where .Result = C {
  fn Op[self: C](other: C) -> C { return {}; }
}
impl forall [T:! Core.ImplicitAs(C)] C as Core.AddAssignWith(T) {
  fn Op[addr self: C*](other: C) {}
}

fn Test(a: C, b: D, z: bool) -> C {
  if (z) {
    return a + a;
  } else {
    // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:12: error: cannot access member of interface `Core.AddWith(D)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
    // CHECK:STDERR:     return a + b;
    // CHECK:STDERR:            ^~~~~
    // CHECK:STDERR:
    return a + b;
  }
}

fn TestAssign(b: D) {
  var a: C = {};

  a += a;

  // CHECK:STDERR: fail_no_impl_for_arg.carbon:[[@LINE+4]]:3: error: cannot access member of interface `Core.AddAssignWith(D)` in type `C` that does not implement that interface [MissingImplInMemberAccess]
  // CHECK:STDERR:   a += b;
  // CHECK:STDERR:   ^~~~~~
  // CHECK:STDERR:
  a += b;
}
