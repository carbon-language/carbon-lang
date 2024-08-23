// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_to_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_to_empty.carbon

// CHECK:STDERR: fail_assign_to_empty.carbon:[[@LINE+3]]:13: ERROR: Cannot initialize tuple of 0 element(s) from tuple with 1 element(s).
// CHECK:STDERR: var x: () = ((),);
// CHECK:STDERR:             ^~~~~
var x: () = ((),);

// CHECK:STDOUT: --- fail_assign_to_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (%.1) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_9.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_9.2: type = converted %.loc14_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_15: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_17: %.2 = tuple_literal (%.loc14_15)
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
