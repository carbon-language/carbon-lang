// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/access.carbon

// --- const_member_access.carbon
library "[[@TEST_NAME]]";

class C {
  var x: ();
}

fn F(c: const C) {
  c.x;
}

// --- fail_const_member_assign.carbon
library "[[@TEST_NAME]]";

class C {
  var x: ();
}

fn F(c: const C) {
  // CHECK:STDERR: fail_const_member_assign.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   c.x = ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  c.x = ();
}

// --- fail_const_ref_member_assign.carbon
library "[[@TEST_NAME]]";

class C {
  var x: ();
}

fn F(ref c: const C) {
  // CHECK:STDERR: fail_const_ref_member_assign.carbon:[[@LINE+4]]:3: error: expression is not assignable [AssignmentToNonAssignable]
  // CHECK:STDERR:   c.x = ();
  // CHECK:STDERR:   ^~~
  // CHECK:STDERR:
  c.x = ();
}
