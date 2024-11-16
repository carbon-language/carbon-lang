// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/struct/member_access.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/struct/member_access.carbon

var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
var y: i32 = x.b;
var z: i32 = y;

// CHECK:STDOUT: --- member_access.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 64 [template]
// CHECK:STDOUT:   %Float.type: type = fn_type @Float [template]
// CHECK:STDOUT:   %Float: %Float.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {.a: f64, .b: i32} [template]
// CHECK:STDOUT:   %.4: f64 = float_literal 0 [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.6: type = struct_type {.a: f64, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.5, %Convert.15 [template]
// CHECK:STDOUT:   %.32: i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct: %.2 = struct_value (%.4, %.32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Float = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
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
// CHECK:STDOUT:     .z = %z
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %.loc11_13.1: Core.IntLiteral = int_value 64 [template = constants.%.1]
// CHECK:STDOUT:   %float.make_type: init type = call constants.%Float(%.loc11_13.1) [template = f64]
// CHECK:STDOUT:   %.loc11_13.2: type = value_of_initializer %float.make_type [template = f64]
// CHECK:STDOUT:   %.loc11_13.3: type = converted %float.make_type, %.loc11_13.2 [template = f64]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_22.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_22.2: type = converted %int.make_type_32.loc11, %.loc11_22.1 [template = i32]
// CHECK:STDOUT:   %.loc11_25: type = struct_type {.a: f64, .b: i32} [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref %.2 = var x
// CHECK:STDOUT:   %x: ref %.2 = bind_name x, %x.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %y.var: ref i32 = var y
// CHECK:STDOUT:   %y: ref i32 = bind_name y, %y.var
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %int.make_type_32.loc13, %.loc13_8.1 [template = i32]
// CHECK:STDOUT:   %z.var: ref i32 = var z
// CHECK:STDOUT:   %z: ref i32 = bind_name z, %z.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_35: f64 = float_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_45: Core.IntLiteral = int_value 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_46.1: %.6 = struct_literal (%.loc11_35, %.loc11_45)
// CHECK:STDOUT:   %.loc11_46.2: ref f64 = struct_access file.%x.var, element0
// CHECK:STDOUT:   %.loc11_46.3: init f64 = initialize_from %.loc11_35 to %.loc11_46.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_46.4: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_46.5: <bound method> = bound_method %.loc11_45, %.loc11_46.4 [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked: init i32 = call %.loc11_46.5(%.loc11_45) [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_46.6: init i32 = converted %.loc11_45, %int.convert_checked [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_46.7: ref i32 = struct_access file.%x.var, element1
// CHECK:STDOUT:   %.loc11_46.8: init i32 = initialize_from %.loc11_46.6 to %.loc11_46.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_46.9: init %.2 = struct_init (%.loc11_46.3, %.loc11_46.8) to file.%x.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc11_47: init %.2 = converted %.loc11_46.1, %.loc11_46.9 [template = constants.%struct]
// CHECK:STDOUT:   assign file.%x.var, %.loc11_47
// CHECK:STDOUT:   %x.ref: ref %.2 = name_ref x, file.%x
// CHECK:STDOUT:   %.loc12_15.1: ref i32 = struct_access %x.ref, element1
// CHECK:STDOUT:   %.loc12_15.2: i32 = bind_value %.loc12_15.1
// CHECK:STDOUT:   assign file.%y.var, %.loc12_15.2
// CHECK:STDOUT:   %y.ref: ref i32 = name_ref y, file.%y
// CHECK:STDOUT:   %.loc13: i32 = bind_value %y.ref
// CHECK:STDOUT:   assign file.%z.var, %.loc13
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
