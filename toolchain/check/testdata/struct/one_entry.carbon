// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/one_entry.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/one_entry.carbon

var x: {.a: i32} = {.a = 4};
var y: {.a: i32} = x;

// CHECK:STDOUT: --- one_entry.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {.a: i32} [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.2, %Convert.15 [template]
// CHECK:STDOUT:   %.29: i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct: %.1 = struct_value (%.29) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .x = %x
// CHECK:STDOUT:     .y = %y
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_13.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_13.2: type = converted %int.make_type_32.loc11, %.loc11_13.1 [template = i32]
// CHECK:STDOUT:   %.loc11_16: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:   %x.var: ref %.1 = var x
// CHECK:STDOUT:   %x: ref %.1 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_13.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_13.2: type = converted %int.make_type_32.loc12, %.loc12_13.1 [template = i32]
// CHECK:STDOUT:   %.loc12_16: type = struct_type {.a: i32} [template = constants.%.1]
// CHECK:STDOUT:   %y.var: ref %.1 = var y
// CHECK:STDOUT:   %y: ref %.1 = bind_name y, %y.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_26: Core.IntLiteral = int_value 4 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_27.1: %.3 = struct_literal (%.loc11_26)
// CHECK:STDOUT:   %.loc11_27.2: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_27.3: <bound method> = bound_method %.loc11_26, %.loc11_27.2 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc11_27.3(%.loc11_26) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.4: init i32 = converted %.loc11_26, %int.convert_checked [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_27.5: init %.1 = struct_init (%.loc11_27.4) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_28: init %.1 = converted %.loc11_27.1, %.loc11_27.5 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_28
// CHECK:STDOUT:   %x.ref: ref %.1 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_20.1: ref i32 = struct_access %x.ref, element0
// CHECK:STDOUT:   %.loc12_20.2: i32 = bind_value %.loc12_20.1
// CHECK:STDOUT:   %.loc12_20.3: init %.1 = struct_init (%.loc12_20.2) to file.%y.var
// CHECK:STDOUT:   %.loc12_21: init %.1 = converted %x.ref, %.loc12_20.3
// CHECK:STDOUT:   assign file.%y.var, %.loc12_21
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
