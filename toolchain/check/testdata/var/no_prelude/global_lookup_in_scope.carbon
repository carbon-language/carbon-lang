// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/var/no_prelude/global_lookup_in_scope.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/var/no_prelude/global_lookup_in_scope.carbon

var x: {.v: ()} = {.v = ()};

fn Main() {
  var y: {.v: ()} = x;
}

// CHECK:STDOUT: --- global_lookup_in_scope.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.v: %.1} [template]
// CHECK:STDOUT:   %tuple: %.1 = tuple_value () [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%tuple) [template]
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11_14.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc11_14.2: type = converted %.loc11_14.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc11_15: type = struct_type {.v: %.1} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc14_16.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc14_16.2: type = converted %.loc14_16.1, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc14_17: type = struct_type {.v: %.1} [template = constants.%.2]
// CHECK:STDOUT:   %y.var: ref %.2 = var y
// CHECK:STDOUT:   %y: ref %.2 = bind_name y, %y.var
// CHECK:STDOUT:   %x.ref: ref %.2 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc14_21.1: ref %.1 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc14_21.2: ref %.1 = struct_access %y.var, element0
// CHECK:STDOUT:   %.loc14_21.3: init %.1 = tuple_init () to %.loc14_21.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21.4: init %.1 = converted %.loc14_21.1, %.loc14_21.3 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc14_21.5: init %.2 = struct_init (%.loc14_21.4) to %y.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc14_22: init %.2 = converted %x.ref, %.loc14_21.5 [template = constants.%struct]
// CHECK:STDOUT:   assign %y.var, %.loc14_22
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26.1: %.1 = tuple_literal ()
// CHECK:STDOUT:   %.loc11_27.1: %.2 = struct_literal (%.loc11_26.1)
// CHECK:STDOUT:   %.loc11_27.2: ref %.1 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_26.2: init %.1 = tuple_init () to %.loc11_27.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_27.3: init %.1 = converted %.loc11_26.1, %.loc11_26.2 [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_27.4: init %.2 = struct_init (%.loc11_27.3) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_28: init %.2 = converted %.loc11_27.1, %.loc11_27.4 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_28
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
