// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/no_prelude/fail_assign_empty.carbon

// CHECK:STDERR: fail_assign_empty.carbon:[[@LINE+3]]:16: error: cannot initialize tuple of 1 element from tuple with 0 elements [TupleInitElementCountMismatch]
// CHECK:STDERR: var x: ((),) = ();
// CHECK:STDERR:                ^~
var x: ((),) = ();

// CHECK:STDOUT: --- fail_assign_empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%empty_tuple.type) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc14_10: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc14_12.1: %tuple.type = tuple_literal (%.loc14_10)
// CHECK:STDOUT:   %.loc14_12.2: type = converted %.loc14_10, constants.%empty_tuple.type [template = constants.%empty_tuple.type]
// CHECK:STDOUT:   %.loc14_12.3: type = converted %.loc14_12.1, constants.%tuple.type [template = constants.%tuple.type]
// CHECK:STDOUT:   %x.var: ref %tuple.type = var x
// CHECK:STDOUT:   %x: ref %tuple.type = bind_name x, %x.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   assign file.%x.var, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
