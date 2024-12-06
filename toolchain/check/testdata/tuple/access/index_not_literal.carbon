// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/tuple/access/index_not_literal.carbon

var a: (bool, i32) = (true, 34);
var b: i32 = a.({.index = 1}.index);
var c: bool = a.(0 as i32);
var d: i32 = a.({.index = 1 as i32}.index);

// CHECK:STDOUT: --- index_not_literal.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [template]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (bool, i32) [template]
// CHECK:STDOUT:   %.2: bool = bool_literal true [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.5: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.9: type = fn_type @Convert.6, @As(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.27: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.28: <bound method> = bound_method %.3, %Convert.15 [template]
// CHECK:STDOUT:   %.29: i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.2, %.29) [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.31: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.1: %.31 = struct_value (%.30) [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.12 [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %.34: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %.35: <bound method> = bound_method %.32, %Convert.16 [template]
// CHECK:STDOUT:   %.36: i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.13 [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %.37: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.36, %Convert.17 [template]
// CHECK:STDOUT:   %.39: <bound method> = bound_method %.30, %Convert.16 [template]
// CHECK:STDOUT:   %.40: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.41: type = struct_type {.index: i32} [template]
// CHECK:STDOUT:   %struct.2: %.41 = struct_value (%.40) [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.40, %Convert.17 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int32 = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     .As = %import_ref.52
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:     .c = %c
// CHECK:STDOUT:     .d = %d
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %bool.make_type.loc11: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_18.1: %tuple.type.1 = tuple_literal (%bool.make_type.loc11, %int.make_type_32.loc11)
// CHECK:STDOUT:   %.loc11_18.2: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:   %.loc11_18.3: type = converted %bool.make_type.loc11, %.loc11_18.2 [template = bool]
// CHECK:STDOUT:   %.loc11_18.4: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_18.5: type = converted %int.make_type_32.loc11, %.loc11_18.4 [template = i32]
// CHECK:STDOUT:   %.loc11_18.6: type = converted %.loc11_18.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_32.loc12, %.loc12_8.1 [template = i32]
// CHECK:STDOUT:   %b.var: ref i32 = var b
// CHECK:STDOUT:   %b: ref i32 = bind_name b, %b.var
// CHECK:STDOUT:   %bool.make_type.loc13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %bool.make_type.loc13 [template = bool]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %bool.make_type.loc13, %.loc13_8.1 [template = bool]
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_32.loc14, %.loc14_8.1 [template = i32]
// CHECK:STDOUT:   %d.var: ref i32 = var d
// CHECK:STDOUT:   %d: ref i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_23: bool = bool_literal true [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_29: Core.IntLiteral = int_value 34 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3 = tuple_literal (%.loc11_23, %.loc11_29)
// CHECK:STDOUT:   %.loc11_31.2: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.3: init bool = initialize_from %.loc11_23 to %.loc11_31.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc11_31.4: %Convert.type.2 = interface_witness_access constants.%.27, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc11_31.5: <bound method> = bound_method %.loc11_29, %.loc11_31.4 [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc11: init i32 = call %.loc11_31.5(%.loc11_29) [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_31.6: init i32 = converted %.loc11_29, %int.convert_checked.loc11 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_31.7: ref i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.8: init i32 = initialize_from %.loc11_31.6 to %.loc11_31.7 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_31.9: init %tuple.type.2 = tuple_init (%.loc11_31.3, %.loc11_31.8) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.2 = converted %.loc11_31.1, %.loc11_31.9 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc12_27: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_28.1: %.31 = struct_literal (%.loc12_27)
// CHECK:STDOUT:   %struct.loc12: %.31 = struct_value (%.loc12_27) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_28.2: %.31 = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%.30]
// CHECK:STDOUT:   %.loc12_15.1: ref i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15.2: i32 = bind_value %.loc12_15.1
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15.2
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc13_18: Core.IntLiteral = int_value 0 [template = constants.%.32]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_23.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_23.2: type = converted %int.make_type_32.loc13, %.loc13_23.1 [template = i32]
// CHECK:STDOUT:   %.loc13_20.1: %Convert.type.9 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc13_20.2: <bound method> = bound_method %.loc13_18, %.loc13_20.1 [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init i32 = call %.loc13_20.2(%.loc13_18) [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_20.3: i32 = value_of_initializer %int.convert_checked.loc13_20.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_20.4: i32 = converted %.loc13_18, %.loc13_20.3 [template = constants.%.36]
// CHECK:STDOUT:   %.loc13_20.5: %Convert.type.5 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc13_20.6: <bound method> = bound_method %.loc13_20.4, %.loc13_20.5 [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %.loc13_20.6(%.loc13_20.4) [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_20.7: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_20.8: Core.IntLiteral = converted %.loc13_20.4, %.loc13_20.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_16.1: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16.2: bool = bind_value %.loc13_16.1
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16.2
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %.loc14_27: Core.IntLiteral = int_value 1 [template = constants.%.30]
// CHECK:STDOUT:   %int.make_type_32.loc14: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc14_32.1: type = value_of_initializer %int.make_type_32.loc14 [template = i32]
// CHECK:STDOUT:   %.loc14_32.2: type = converted %int.make_type_32.loc14, %.loc14_32.1 [template = i32]
// CHECK:STDOUT:   %.loc14_29.1: %Convert.type.9 = interface_witness_access constants.%.34, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %.loc14_29.2: <bound method> = bound_method %.loc14_27, %.loc14_29.1 [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init i32 = call %.loc14_29.2(%.loc14_27) [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_29.3: i32 = value_of_initializer %int.convert_checked.loc14_29 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_29.4: i32 = converted %.loc14_27, %.loc14_29.3 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_35.1: %.41 = struct_literal (%.loc14_29.4)
// CHECK:STDOUT:   %struct.loc14: %.41 = struct_value (%.loc14_29.4) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_35.2: %.41 = converted %.loc14_35.1, %struct.loc14 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_36.1: i32 = struct_access %.loc14_35.2, element0 [template = constants.%.40]
// CHECK:STDOUT:   %.loc14_36.2: %Convert.type.5 = interface_witness_access constants.%.37, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %.loc14_36.3: <bound method> = bound_method %.loc14_36.1, %.loc14_36.2 [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %.loc14_36.3(%.loc14_36.1) [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_36.4: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_36.5: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.4 [template = constants.%.30]
// CHECK:STDOUT:   %.loc14_15.1: ref i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15.2: i32 = bind_value %.loc14_15.1
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
