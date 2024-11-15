// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/operators/builtin/assignment.carbon

fn Main() {
  var a: i32 = 12;
  a = 9;

  var b: (i32, i32) = (1, 2);
  b.0 = 3;
  b.1 = 4;

  var c: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  c.a = 3;
  c.b = 4;

  var p: i32* = &a;
  *p = 5;

  *(if true then p else &a) = 10;
}

// CHECK:STDOUT: --- assignment.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Main.type: type = fn_type @Main [template]
// CHECK:STDOUT:   %Main: %Main.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(i32) [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.11 [template]
// CHECK:STDOUT:   %Convert.15: %Convert.type.15 = struct_value () [template]
// CHECK:STDOUT:   %.25: <witness> = interface_witness (%Convert.15) [template]
// CHECK:STDOUT:   %.26: <bound method> = bound_method %.1, %Convert.15 [template]
// CHECK:STDOUT:   %.27: i32 = int_value 12 [template]
// CHECK:STDOUT:   %.28: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.28, %Convert.15 [template]
// CHECK:STDOUT:   %.30: i32 = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.32: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.33: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.32, %Convert.15 [template]
// CHECK:STDOUT:   %.35: i32 = int_value 1 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %.33, %Convert.15 [template]
// CHECK:STDOUT:   %.37: i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.35, %.37) [template]
// CHECK:STDOUT:   %.38: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.39: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.39, %Convert.15 [template]
// CHECK:STDOUT:   %.41: i32 = int_value 3 [template]
// CHECK:STDOUT:   %.42: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %.42, %Convert.15 [template]
// CHECK:STDOUT:   %.44: i32 = int_value 4 [template]
// CHECK:STDOUT:   %.45: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.47: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %.45 = struct_value (%.35, %.37) [template]
// CHECK:STDOUT:   %.48: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.49: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.50: <bound method> = bound_method %.49, %Convert.15 [template]
// CHECK:STDOUT:   %.51: i32 = int_value 5 [template]
// CHECK:STDOUT:   %.52: bool = bool_literal true [template]
// CHECK:STDOUT:   %.53: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.54: <bound method> = bound_method %.53, %Convert.15 [template]
// CHECK:STDOUT:   %.55: i32 = int_value 10 [template]
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
// CHECK:STDOUT:     .Main = %Main.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Main.decl: %Main.type = fn_decl @Main [template = constants.%Main] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Main() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int.make_type_32.loc12: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_32.loc12 [template = i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_32.loc12, %.loc12_10.1 [template = i32]
// CHECK:STDOUT:   %a.var: ref i32 = var a
// CHECK:STDOUT:   %a: ref i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16: Core.IntLiteral = int_value 12 [template = constants.%.1]
// CHECK:STDOUT:   %.loc12_18.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc12_18.2: <bound method> = bound_method %.loc12_16, %.loc12_18.1 [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc12: init i32 = call %.loc12_18.2(%.loc12_16) [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_18.3: init i32 = converted %.loc12_16, %int.convert_checked.loc12 [template = constants.%.27]
// CHECK:STDOUT:   assign %a.var, %.loc12_18.3
// CHECK:STDOUT:   %a.ref.loc13: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc13_7: Core.IntLiteral = int_value 9 [template = constants.%.28]
// CHECK:STDOUT:   %.loc13_5.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc13_5.2: <bound method> = bound_method %.loc13_7, %.loc13_5.1 [template = constants.%.29]
// CHECK:STDOUT:   %int.convert_checked.loc13: init i32 = call %.loc13_5.2(%.loc13_7) [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_5.3: init i32 = converted %.loc13_7, %int.convert_checked.loc13 [template = constants.%.30]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13_5.3
// CHECK:STDOUT:   %int.make_type_32.loc15_11: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc15_16: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_19.1: %tuple.type.1 = tuple_literal (%int.make_type_32.loc15_11, %int.make_type_32.loc15_16)
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_32.loc15_11 [template = i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_32.loc15_11, %.loc15_19.2 [template = i32]
// CHECK:STDOUT:   %.loc15_19.4: type = value_of_initializer %int.make_type_32.loc15_16 [template = i32]
// CHECK:STDOUT:   %.loc15_19.5: type = converted %int.make_type_32.loc15_16, %.loc15_19.4 [template = i32]
// CHECK:STDOUT:   %.loc15_19.6: type = converted %.loc15_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc15_24: Core.IntLiteral = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc15_27: Core.IntLiteral = int_value 2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.3 = tuple_literal (%.loc15_24, %.loc15_27)
// CHECK:STDOUT:   %.loc15_28.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_28.3: <bound method> = bound_method %.loc15_24, %.loc15_28.2 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.1: init i32 = call %.loc15_28.3(%.loc15_24) [template = constants.%.35]
// CHECK:STDOUT:   %.loc15_28.4: init i32 = converted %.loc15_24, %int.convert_checked.loc15_28.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc15_28.5: ref i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.6: init i32 = initialize_from %.loc15_28.4 to %.loc15_28.5 [template = constants.%.35]
// CHECK:STDOUT:   %.loc15_28.7: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc15_28.8: <bound method> = bound_method %.loc15_27, %.loc15_28.7 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.2: init i32 = call %.loc15_28.8(%.loc15_27) [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_28.9: init i32 = converted %.loc15_27, %int.convert_checked.loc15_28.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_28.10: ref i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.11: init i32 = initialize_from %.loc15_28.9 to %.loc15_28.10 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_28.12: init %tuple.type.2 = tuple_init (%.loc15_28.6, %.loc15_28.11) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_29: init %tuple.type.2 = converted %.loc15_28.1, %.loc15_28.12 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_29
// CHECK:STDOUT:   %b.ref.loc16: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc16_5: Core.IntLiteral = int_value 0 [template = constants.%.38]
// CHECK:STDOUT:   %.loc16_4: ref i32 = tuple_access %b.ref.loc16, element0
// CHECK:STDOUT:   %.loc16_9: Core.IntLiteral = int_value 3 [template = constants.%.39]
// CHECK:STDOUT:   %.loc16_7.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc16_7.2: <bound method> = bound_method %.loc16_9, %.loc16_7.1 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc16: init i32 = call %.loc16_7.2(%.loc16_9) [template = constants.%.41]
// CHECK:STDOUT:   %.loc16_7.3: init i32 = converted %.loc16_9, %int.convert_checked.loc16 [template = constants.%.41]
// CHECK:STDOUT:   assign %.loc16_4, %.loc16_7.3
// CHECK:STDOUT:   %b.ref.loc17: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc17_5: Core.IntLiteral = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc17_4: ref i32 = tuple_access %b.ref.loc17, element1
// CHECK:STDOUT:   %.loc17_9: Core.IntLiteral = int_value 4 [template = constants.%.42]
// CHECK:STDOUT:   %.loc17_7.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc17_7.2: <bound method> = bound_method %.loc17_9, %.loc17_7.1 [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc17: init i32 = call %.loc17_7.2(%.loc17_9) [template = constants.%.44]
// CHECK:STDOUT:   %.loc17_7.3: init i32 = converted %.loc17_9, %int.convert_checked.loc17 [template = constants.%.44]
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_7.3
// CHECK:STDOUT:   %int.make_type_32.loc19_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_15.1: type = value_of_initializer %int.make_type_32.loc19_15 [template = i32]
// CHECK:STDOUT:   %.loc19_15.2: type = converted %int.make_type_32.loc19_15, %.loc19_15.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc19_24: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_24.1: type = value_of_initializer %int.make_type_32.loc19_24 [template = i32]
// CHECK:STDOUT:   %.loc19_24.2: type = converted %int.make_type_32.loc19_24, %.loc19_24.1 [template = i32]
// CHECK:STDOUT:   %.loc19_27: type = struct_type {.a: i32, .b: i32} [template = constants.%.45]
// CHECK:STDOUT:   %c.var: ref %.45 = var c
// CHECK:STDOUT:   %c: ref %.45 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc19_37: Core.IntLiteral = int_value 1 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_45: Core.IntLiteral = int_value 2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc19_46.1: %.47 = struct_literal (%.loc19_37, %.loc19_45)
// CHECK:STDOUT:   %.loc19_46.2: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc19_46.3: <bound method> = bound_method %.loc19_37, %.loc19_46.2 [template = constants.%.34]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.1: init i32 = call %.loc19_46.3(%.loc19_37) [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_46.4: init i32 = converted %.loc19_37, %int.convert_checked.loc19_46.1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_46.5: ref i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc19_46.6: init i32 = initialize_from %.loc19_46.4 to %.loc19_46.5 [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_46.7: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc19_46.8: <bound method> = bound_method %.loc19_45, %.loc19_46.7 [template = constants.%.36]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.2: init i32 = call %.loc19_46.8(%.loc19_45) [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_46.9: init i32 = converted %.loc19_45, %int.convert_checked.loc19_46.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_46.10: ref i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc19_46.11: init i32 = initialize_from %.loc19_46.9 to %.loc19_46.10 [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_46.12: init %.45 = struct_init (%.loc19_46.6, %.loc19_46.11) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_47: init %.45 = converted %.loc19_46.1, %.loc19_46.12 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc19_47
// CHECK:STDOUT:   %c.ref.loc20: ref %.45 = name_ref c, %c
// CHECK:STDOUT:   %.loc20_4: ref i32 = struct_access %c.ref.loc20, element0
// CHECK:STDOUT:   %.loc20_9: Core.IntLiteral = int_value 3 [template = constants.%.39]
// CHECK:STDOUT:   %.loc20_7.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc20_7.2: <bound method> = bound_method %.loc20_9, %.loc20_7.1 [template = constants.%.40]
// CHECK:STDOUT:   %int.convert_checked.loc20: init i32 = call %.loc20_7.2(%.loc20_9) [template = constants.%.41]
// CHECK:STDOUT:   %.loc20_7.3: init i32 = converted %.loc20_9, %int.convert_checked.loc20 [template = constants.%.41]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_7.3
// CHECK:STDOUT:   %c.ref.loc21: ref %.45 = name_ref c, %c
// CHECK:STDOUT:   %.loc21_4: ref i32 = struct_access %c.ref.loc21, element1
// CHECK:STDOUT:   %.loc21_9: Core.IntLiteral = int_value 4 [template = constants.%.42]
// CHECK:STDOUT:   %.loc21_7.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc21_7.2: <bound method> = bound_method %.loc21_9, %.loc21_7.1 [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc21: init i32 = call %.loc21_7.2(%.loc21_9) [template = constants.%.44]
// CHECK:STDOUT:   %.loc21_7.3: init i32 = converted %.loc21_9, %int.convert_checked.loc21 [template = constants.%.44]
// CHECK:STDOUT:   assign %.loc21_4, %.loc21_7.3
// CHECK:STDOUT:   %int.make_type_32.loc23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc23_13.1: type = value_of_initializer %int.make_type_32.loc23 [template = i32]
// CHECK:STDOUT:   %.loc23_13.2: type = converted %int.make_type_32.loc23, %.loc23_13.1 [template = i32]
// CHECK:STDOUT:   %.loc23_13.3: type = ptr_type i32 [template = constants.%.48]
// CHECK:STDOUT:   %p.var: ref %.48 = var p
// CHECK:STDOUT:   %p: ref %.48 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc23: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc23_17: %.48 = addr_of %a.ref.loc23
// CHECK:STDOUT:   assign %p.var, %.loc23_17
// CHECK:STDOUT:   %p.ref.loc24: ref %.48 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_4: %.48 = bind_value %p.ref.loc24
// CHECK:STDOUT:   %.loc24_3: ref i32 = deref %.loc24_4
// CHECK:STDOUT:   %.loc24_8: Core.IntLiteral = int_value 5 [template = constants.%.49]
// CHECK:STDOUT:   %.loc24_6.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc24_6.2: <bound method> = bound_method %.loc24_8, %.loc24_6.1 [template = constants.%.50]
// CHECK:STDOUT:   %int.convert_checked.loc24: init i32 = call %.loc24_6.2(%.loc24_8) [template = constants.%.51]
// CHECK:STDOUT:   %.loc24_6.3: init i32 = converted %.loc24_8, %int.convert_checked.loc24 [template = constants.%.51]
// CHECK:STDOUT:   assign %.loc24_3, %.loc24_6.3
// CHECK:STDOUT:   %.loc26_8: bool = bool_literal true [template = constants.%.52]
// CHECK:STDOUT:   if %.loc26_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc26: ref %.48 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_18: %.48 = bind_value %p.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc26: ref i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_25: %.48 = addr_of %a.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_5: %.48 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc26_3: ref i32 = deref %.loc26_5
// CHECK:STDOUT:   %.loc26_31: Core.IntLiteral = int_value 10 [template = constants.%.53]
// CHECK:STDOUT:   %.loc26_29.1: %Convert.type.2 = interface_witness_access constants.%.25, element0 [template = constants.%Convert.15]
// CHECK:STDOUT:   %.loc26_29.2: <bound method> = bound_method %.loc26_31, %.loc26_29.1 [template = constants.%.54]
// CHECK:STDOUT:   %int.convert_checked.loc26: init i32 = call %.loc26_29.2(%.loc26_31) [template = constants.%.55]
// CHECK:STDOUT:   %.loc26_29.3: init i32 = converted %.loc26_31, %int.convert_checked.loc26 [template = constants.%.55]
// CHECK:STDOUT:   assign %.loc26_3, %.loc26_29.3
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
