// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/no_prelude/empty.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/no_prelude/empty.carbon

var x: {} = {};
var y: {} = x;

// CHECK:STDOUT: --- empty.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_9.2: type = converted %.loc11_9.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %x.var: ref %empty_struct_type = var x
// CHECK:STDOUT:   %x: ref %empty_struct_type = bind_name x, %x.var
// CHECK:STDOUT:   %.loc12_9.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_9.2: type = converted %.loc12_9.1, constants.%empty_struct_type [template = constants.%empty_struct_type]
// CHECK:STDOUT:   %y.var: ref %empty_struct_type = var y
// CHECK:STDOUT:   %y: ref %empty_struct_type = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_14.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc11_14.2: init %empty_struct_type = struct_init () to file.%x.var [template = constants.%empty_struct]
// CHECK:STDOUT:   %.loc11_15: init %empty_struct_type = converted %.loc11_14.1, %.loc11_14.2 [template = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_15
// CHECK:STDOUT:   %x.ref: ref %empty_struct_type = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_13: init %empty_struct_type = struct_init () to file.%y.var [template = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_14: init %empty_struct_type = converted %x.ref, %.loc12_13 [template = constants.%empty_struct]
// CHECK:STDOUT:   assign file.%y.var, %.loc12_14
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
