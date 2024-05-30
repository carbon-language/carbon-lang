// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuples/no_prelude/empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/tuples/no_prelude/empty.carbon

var x: () = ();
var y: () = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %tuple: () = tuple_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref () = var x
// CHECK:STDOUT:   %x: ref () = bind_name x, %x.var
// CHECK:STDOUT:   %.loc12_9.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref () = var y
// CHECK:STDOUT:   %y: ref () = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14.1: () = tuple_literal ()
// CHECK:STDOUT:   %.loc11_14.2: init () = tuple_init () to file.%x.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_15: init () = converted %.loc11_14.1, %.loc11_14.2 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_15
// CHECK:STDOUT:   %x.ref: ref () = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_13: init () = tuple_init () to file.%y.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc12_14: init () = converted %x.ref, %.loc12_13 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%y.var, %.loc12_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
