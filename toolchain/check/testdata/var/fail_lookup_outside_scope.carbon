// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/fail_lookup_outside_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/fail_lookup_outside_scope.carbon

// --- fail_basic.carbon

fn Main() {
  var unused x: ();
}

// CHECK:STDERR: fail_basic.carbon:[[@LINE+4]]:13: error: name `x` not found [NameNotFound]
// CHECK:STDERR: var y: () = x;
// CHECK:STDERR:             ^
// CHECK:STDERR:
var y: () = x;
