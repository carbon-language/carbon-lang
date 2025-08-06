// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_unknown_member.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_unknown_member.carbon

class Class {
  var n: i32;
}

fn G(c: Class) -> i32 {
  // TODO: Mention the scope in which we looked for the name.
  // CHECK:STDERR: fail_unknown_member.carbon:[[@LINE+4]]:10: error: member name `something` not found in `Class` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   return c.something;
  // CHECK:STDERR:          ^~~~~~~~~~~
  // CHECK:STDERR:
  return c.something;
}
