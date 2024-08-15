// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/class/value_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/class/value_access.carbon

class C {
  var a: (i32, i32, i32);
}

fn F(c: C) -> i32 {
  // TODO: `c.a` is a value expression here, which forces a value binding as
  // part of the member access, creating a tuple value temporary. We could
  // defer performing the value binding to avoid creating this temporary.
  // CHECK:STDERR: value_access.carbon:[[@LINE+3]]:10: ERROR: Type `(i32, i32, i32)` does not support indexing.
  // CHECK:STDERR:   return c.a[1];
  // CHECK:STDERR:          ^~~~~~
  return c.a[1];
}
