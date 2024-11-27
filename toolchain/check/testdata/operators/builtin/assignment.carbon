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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.26: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %.2, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.29: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %.30: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.30, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.35: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.35, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %.36, %Convert.14 [template]
// CHECK:STDOUT:   %.41: <specific function> = specific_function %.40, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.42: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%.39, %.42) [template]
// CHECK:STDOUT:   %.43: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %.44: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.45: <bound method> = bound_method %.44, %Convert.14 [template]
// CHECK:STDOUT:   %.46: <specific function> = specific_function %.45, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.47: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.48: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.49: <bound method> = bound_method %.48, %Convert.14 [template]
// CHECK:STDOUT:   %.50: <specific function> = specific_function %.49, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.51: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %.52: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %.54: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %.52 = struct_value (%.39, %.42) [template]
// CHECK:STDOUT:   %.55: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %.56: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.57: <bound method> = bound_method %.56, %Convert.14 [template]
// CHECK:STDOUT:   %.58: <specific function> = specific_function %.57, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.59: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.60: bool = bool_literal true [template]
// CHECK:STDOUT:   %.61: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.62: <bound method> = bound_method %.61, %Convert.14 [template]
// CHECK:STDOUT:   %.63: <specific function> = specific_function %.62, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.64: %i32 = int_value 10 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
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
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed.loc12, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %.loc12_16: Core.IntLiteral = int_value 12 [template = constants.%.2]
// CHECK:STDOUT:   %.loc12_18.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_18.2: <bound method> = bound_method %.loc12_16, %.loc12_18.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc12_18.3: <specific function> = specific_function %.loc12_18.2, @Convert.2(constants.%.1) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_18.3(%.loc12_16) [template = constants.%.29]
// CHECK:STDOUT:   %.loc12_18.4: init %i32 = converted %.loc12_16, %int.convert_checked.loc12 [template = constants.%.29]
// CHECK:STDOUT:   assign %a.var, %.loc12_18.4
// CHECK:STDOUT:   %a.ref.loc13: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc13_7: Core.IntLiteral = int_value 9 [template = constants.%.30]
// CHECK:STDOUT:   %.loc13_5.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_5.2: <bound method> = bound_method %.loc13_7, %.loc13_5.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_5.3: <specific function> = specific_function %.loc13_5.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %.loc13_5.3(%.loc13_7) [template = constants.%.33]
// CHECK:STDOUT:   %.loc13_5.4: init %i32 = converted %.loc13_7, %int.convert_checked.loc13 [template = constants.%.33]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13_5.4
// CHECK:STDOUT:   %.loc15_11: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_11: init type = call constants.%Int(%.loc15_11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_16: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15_16: init type = call constants.%Int(%.loc15_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc15_11, %int.make_type_signed.loc15_16)
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_signed.loc15_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_signed.loc15_11, %.loc15_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.4: type = value_of_initializer %int.make_type_signed.loc15_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.5: type = converted %int.make_type_signed.loc15_16, %.loc15_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.6: type = converted %.loc15_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %.loc15_24: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc15_27: Core.IntLiteral = int_value 2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.3 = tuple_literal (%.loc15_24, %.loc15_27)
// CHECK:STDOUT:   %.loc15_28.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.3: <bound method> = bound_method %.loc15_24, %.loc15_28.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc15_28.4: <specific function> = specific_function %.loc15_28.3, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.1: init %i32 = call %.loc15_28.4(%.loc15_24) [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_28.5: init %i32 = converted %.loc15_24, %int.convert_checked.loc15_28.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_28.6: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.7: init %i32 = initialize_from %.loc15_28.5 to %.loc15_28.6 [template = constants.%.39]
// CHECK:STDOUT:   %.loc15_28.8: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.9: <bound method> = bound_method %.loc15_27, %.loc15_28.8 [template = constants.%.40]
// CHECK:STDOUT:   %.loc15_28.10: <specific function> = specific_function %.loc15_28.9, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.2: init %i32 = call %.loc15_28.10(%.loc15_27) [template = constants.%.42]
// CHECK:STDOUT:   %.loc15_28.11: init %i32 = converted %.loc15_27, %int.convert_checked.loc15_28.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc15_28.12: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.13: init %i32 = initialize_from %.loc15_28.11 to %.loc15_28.12 [template = constants.%.42]
// CHECK:STDOUT:   %.loc15_28.14: init %tuple.type.2 = tuple_init (%.loc15_28.7, %.loc15_28.13) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_29: init %tuple.type.2 = converted %.loc15_28.1, %.loc15_28.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_29
// CHECK:STDOUT:   %b.ref.loc16: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc16_5: Core.IntLiteral = int_value 0 [template = constants.%.43]
// CHECK:STDOUT:   %.loc16_4: ref %i32 = tuple_access %b.ref.loc16, element0
// CHECK:STDOUT:   %.loc16_9: Core.IntLiteral = int_value 3 [template = constants.%.44]
// CHECK:STDOUT:   %.loc16_7.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_7.2: <bound method> = bound_method %.loc16_9, %.loc16_7.1 [template = constants.%.45]
// CHECK:STDOUT:   %.loc16_7.3: <specific function> = specific_function %.loc16_7.2, @Convert.2(constants.%.1) [template = constants.%.46]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %.loc16_7.3(%.loc16_9) [template = constants.%.47]
// CHECK:STDOUT:   %.loc16_7.4: init %i32 = converted %.loc16_9, %int.convert_checked.loc16 [template = constants.%.47]
// CHECK:STDOUT:   assign %.loc16_4, %.loc16_7.4
// CHECK:STDOUT:   %b.ref.loc17: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %.loc17_5: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc17_4: ref %i32 = tuple_access %b.ref.loc17, element1
// CHECK:STDOUT:   %.loc17_9: Core.IntLiteral = int_value 4 [template = constants.%.48]
// CHECK:STDOUT:   %.loc17_7.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_7.2: <bound method> = bound_method %.loc17_9, %.loc17_7.1 [template = constants.%.49]
// CHECK:STDOUT:   %.loc17_7.3: <specific function> = specific_function %.loc17_7.2, @Convert.2(constants.%.1) [template = constants.%.50]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %.loc17_7.3(%.loc17_9) [template = constants.%.51]
// CHECK:STDOUT:   %.loc17_7.4: init %i32 = converted %.loc17_9, %int.convert_checked.loc17 [template = constants.%.51]
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_7.4
// CHECK:STDOUT:   %.loc19_15.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19_15: init type = call constants.%Int(%.loc19_15.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.2: type = value_of_initializer %int.make_type_signed.loc19_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.3: type = converted %int.make_type_signed.loc19_15, %.loc19_15.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc19_24: init type = call constants.%Int(%.loc19_24.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.2: type = value_of_initializer %int.make_type_signed.loc19_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.3: type = converted %int.make_type_signed.loc19_24, %.loc19_24.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_27: type = struct_type {.a: %i32, .b: %i32} [template = constants.%.52]
// CHECK:STDOUT:   %c.var: ref %.52 = var c
// CHECK:STDOUT:   %c: ref %.52 = bind_name c, %c.var
// CHECK:STDOUT:   %.loc19_37: Core.IntLiteral = int_value 1 [template = constants.%.35]
// CHECK:STDOUT:   %.loc19_45: Core.IntLiteral = int_value 2 [template = constants.%.36]
// CHECK:STDOUT:   %.loc19_46.1: %.54 = struct_literal (%.loc19_37, %.loc19_45)
// CHECK:STDOUT:   %.loc19_46.2: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_46.3: <bound method> = bound_method %.loc19_37, %.loc19_46.2 [template = constants.%.37]
// CHECK:STDOUT:   %.loc19_46.4: <specific function> = specific_function %.loc19_46.3, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.1: init %i32 = call %.loc19_46.4(%.loc19_37) [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_46.5: init %i32 = converted %.loc19_37, %int.convert_checked.loc19_46.1 [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_46.6: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc19_46.7: init %i32 = initialize_from %.loc19_46.5 to %.loc19_46.6 [template = constants.%.39]
// CHECK:STDOUT:   %.loc19_46.8: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_46.9: <bound method> = bound_method %.loc19_45, %.loc19_46.8 [template = constants.%.40]
// CHECK:STDOUT:   %.loc19_46.10: <specific function> = specific_function %.loc19_46.9, @Convert.2(constants.%.1) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.2: init %i32 = call %.loc19_46.10(%.loc19_45) [template = constants.%.42]
// CHECK:STDOUT:   %.loc19_46.11: init %i32 = converted %.loc19_45, %int.convert_checked.loc19_46.2 [template = constants.%.42]
// CHECK:STDOUT:   %.loc19_46.12: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc19_46.13: init %i32 = initialize_from %.loc19_46.11 to %.loc19_46.12 [template = constants.%.42]
// CHECK:STDOUT:   %.loc19_46.14: init %.52 = struct_init (%.loc19_46.7, %.loc19_46.13) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_47: init %.52 = converted %.loc19_46.1, %.loc19_46.14 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc19_47
// CHECK:STDOUT:   %c.ref.loc20: ref %.52 = name_ref c, %c
// CHECK:STDOUT:   %.loc20_4: ref %i32 = struct_access %c.ref.loc20, element0
// CHECK:STDOUT:   %.loc20_9: Core.IntLiteral = int_value 3 [template = constants.%.44]
// CHECK:STDOUT:   %.loc20_7.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_7.2: <bound method> = bound_method %.loc20_9, %.loc20_7.1 [template = constants.%.45]
// CHECK:STDOUT:   %.loc20_7.3: <specific function> = specific_function %.loc20_7.2, @Convert.2(constants.%.1) [template = constants.%.46]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %.loc20_7.3(%.loc20_9) [template = constants.%.47]
// CHECK:STDOUT:   %.loc20_7.4: init %i32 = converted %.loc20_9, %int.convert_checked.loc20 [template = constants.%.47]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_7.4
// CHECK:STDOUT:   %c.ref.loc21: ref %.52 = name_ref c, %c
// CHECK:STDOUT:   %.loc21_4: ref %i32 = struct_access %c.ref.loc21, element1
// CHECK:STDOUT:   %.loc21_9: Core.IntLiteral = int_value 4 [template = constants.%.48]
// CHECK:STDOUT:   %.loc21_7.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_7.2: <bound method> = bound_method %.loc21_9, %.loc21_7.1 [template = constants.%.49]
// CHECK:STDOUT:   %.loc21_7.3: <specific function> = specific_function %.loc21_7.2, @Convert.2(constants.%.1) [template = constants.%.50]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %.loc21_7.3(%.loc21_9) [template = constants.%.51]
// CHECK:STDOUT:   %.loc21_7.4: init %i32 = converted %.loc21_9, %int.convert_checked.loc21 [template = constants.%.51]
// CHECK:STDOUT:   assign %.loc21_4, %.loc21_7.4
// CHECK:STDOUT:   %.loc23_10: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%.loc23_10) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.1: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.2: type = converted %int.make_type_signed.loc23, %.loc23_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.3: type = ptr_type %i32 [template = constants.%.55]
// CHECK:STDOUT:   %p.var: ref %.55 = var p
// CHECK:STDOUT:   %p: ref %.55 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc23: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc23_17: %.55 = addr_of %a.ref.loc23
// CHECK:STDOUT:   assign %p.var, %.loc23_17
// CHECK:STDOUT:   %p.ref.loc24: ref %.55 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_4: %.55 = bind_value %p.ref.loc24
// CHECK:STDOUT:   %.loc24_3: ref %i32 = deref %.loc24_4
// CHECK:STDOUT:   %.loc24_8: Core.IntLiteral = int_value 5 [template = constants.%.56]
// CHECK:STDOUT:   %.loc24_6.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_6.2: <bound method> = bound_method %.loc24_8, %.loc24_6.1 [template = constants.%.57]
// CHECK:STDOUT:   %.loc24_6.3: <specific function> = specific_function %.loc24_6.2, @Convert.2(constants.%.1) [template = constants.%.58]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %.loc24_6.3(%.loc24_8) [template = constants.%.59]
// CHECK:STDOUT:   %.loc24_6.4: init %i32 = converted %.loc24_8, %int.convert_checked.loc24 [template = constants.%.59]
// CHECK:STDOUT:   assign %.loc24_3, %.loc24_6.4
// CHECK:STDOUT:   %.loc26_8: bool = bool_literal true [template = constants.%.60]
// CHECK:STDOUT:   if %.loc26_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc26: ref %.55 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_18: %.55 = bind_value %p.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc26: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_25: %.55 = addr_of %a.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_5: %.55 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc26_3: ref %i32 = deref %.loc26_5
// CHECK:STDOUT:   %.loc26_31: Core.IntLiteral = int_value 10 [template = constants.%.61]
// CHECK:STDOUT:   %.loc26_29.1: %Convert.type.2 = interface_witness_access constants.%.26, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_29.2: <bound method> = bound_method %.loc26_31, %.loc26_29.1 [template = constants.%.62]
// CHECK:STDOUT:   %.loc26_29.3: <specific function> = specific_function %.loc26_29.2, @Convert.2(constants.%.1) [template = constants.%.63]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %.loc26_29.3(%.loc26_31) [template = constants.%.64]
// CHECK:STDOUT:   %.loc26_29.4: init %i32 = converted %.loc26_31, %int.convert_checked.loc26 [template = constants.%.64]
// CHECK:STDOUT:   assign %.loc26_3, %.loc26_29.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
