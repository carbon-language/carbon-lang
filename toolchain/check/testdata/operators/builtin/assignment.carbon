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
// CHECK:STDOUT:   %int.32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int.32 [template]
// CHECK:STDOUT:   %int.12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int.32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.24: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.25: <bound method> = bound_method %int.12.1, %Convert.14 [template]
// CHECK:STDOUT:   %.26: <specific function> = specific_function %.25, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %int.9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %.27: <bound method> = bound_method %int.9.1, %Convert.14 [template]
// CHECK:STDOUT:   %.28: <specific function> = specific_function %.27, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int.1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int.2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %int.1.1, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %int.2.1, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int.1.2, %int.2.2) [template]
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int.3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %int.3.1, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int.4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.36: <bound method> = bound_method %int.4.1, %Convert.14 [template]
// CHECK:STDOUT:   %.37: <specific function> = specific_function %.36, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int.1.2, %int.2.2) [template]
// CHECK:STDOUT:   %.39: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %int.5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.40: <bound method> = bound_method %int.5.1, %Convert.14 [template]
// CHECK:STDOUT:   %.41: <specific function> = specific_function %.40, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.42: bool = bool_literal true [template]
// CHECK:STDOUT:   %int.10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %.43: <bound method> = bound_method %int.10.1, %Convert.14 [template]
// CHECK:STDOUT:   %.44: <specific function> = specific_function %.43, @Convert.2(%int.32) [template]
// CHECK:STDOUT:   %int.10.2: %i32 = int_value 10 [template]
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
// CHECK:STDOUT:   %int.32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int.32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %int.12: Core.IntLiteral = int_value 12 [template = constants.%int.12.1]
// CHECK:STDOUT:   %.loc12_18.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_18.2: <bound method> = bound_method %int.12, %.loc12_18.1 [template = constants.%.25]
// CHECK:STDOUT:   %.loc12_18.3: <specific function> = specific_function %.loc12_18.2, @Convert.2(constants.%int.32) [template = constants.%.26]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_18.3(%int.12) [template = constants.%int.12.2]
// CHECK:STDOUT:   %.loc12_18.4: init %i32 = converted %int.12, %int.convert_checked.loc12 [template = constants.%int.12.2]
// CHECK:STDOUT:   assign %a.var, %.loc12_18.4
// CHECK:STDOUT:   %a.ref.loc13: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %int.9: Core.IntLiteral = int_value 9 [template = constants.%int.9.1]
// CHECK:STDOUT:   %.loc13_5.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_5.2: <bound method> = bound_method %int.9, %.loc13_5.1 [template = constants.%.27]
// CHECK:STDOUT:   %.loc13_5.3: <specific function> = specific_function %.loc13_5.2, @Convert.2(constants.%int.32) [template = constants.%.28]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %.loc13_5.3(%int.9) [template = constants.%int.9.2]
// CHECK:STDOUT:   %.loc13_5.4: init %i32 = converted %int.9, %int.convert_checked.loc13 [template = constants.%int.9.2]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13_5.4
// CHECK:STDOUT:   %int.32.loc15_11: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_11: init type = call constants.%Int(%int.32.loc15_11) [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc15_16: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_16: init type = call constants.%Int(%int.32.loc15_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc15_11, %int.make_type_signed.loc15_16)
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_signed.loc15_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_signed.loc15_11, %.loc15_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.4: type = value_of_initializer %int.make_type_signed.loc15_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.5: type = converted %int.make_type_signed.loc15_16, %.loc15_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.6: type = converted %.loc15_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %int.1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc15: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.3 = tuple_literal (%int.1.loc15, %int.2.loc15)
// CHECK:STDOUT:   %.loc15_28.2: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.3: <bound method> = bound_method %int.1.loc15, %.loc15_28.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc15_28.4: <specific function> = specific_function %.loc15_28.3, @Convert.2(constants.%int.32) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.1: init %i32 = call %.loc15_28.4(%int.1.loc15) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc15_28.5: init %i32 = converted %int.1.loc15, %int.convert_checked.loc15_28.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc15_28.6: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.7: init %i32 = initialize_from %.loc15_28.5 to %.loc15_28.6 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc15_28.8: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_28.9: <bound method> = bound_method %int.2.loc15, %.loc15_28.8 [template = constants.%.32]
// CHECK:STDOUT:   %.loc15_28.10: <specific function> = specific_function %.loc15_28.9, @Convert.2(constants.%int.32) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.2: init %i32 = call %.loc15_28.10(%int.2.loc15) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc15_28.11: init %i32 = converted %int.2.loc15, %int.convert_checked.loc15_28.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc15_28.12: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.13: init %i32 = initialize_from %.loc15_28.11 to %.loc15_28.12 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc15_28.14: init %tuple.type.2 = tuple_init (%.loc15_28.7, %.loc15_28.13) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_29: init %tuple.type.2 = converted %.loc15_28.1, %.loc15_28.14 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_29
// CHECK:STDOUT:   %b.ref.loc16: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %int.0: Core.IntLiteral = int_value 0 [template = constants.%int.0]
// CHECK:STDOUT:   %.loc16_4: ref %i32 = tuple_access %b.ref.loc16, element0
// CHECK:STDOUT:   %int.3.loc16: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc16_7.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc16_7.2: <bound method> = bound_method %int.3.loc16, %.loc16_7.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc16_7.3: <specific function> = specific_function %.loc16_7.2, @Convert.2(constants.%int.32) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %.loc16_7.3(%int.3.loc16) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc16_7.4: init %i32 = converted %int.3.loc16, %int.convert_checked.loc16 [template = constants.%int.3.2]
// CHECK:STDOUT:   assign %.loc16_4, %.loc16_7.4
// CHECK:STDOUT:   %b.ref.loc17: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %int.1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %.loc17_4: ref %i32 = tuple_access %b.ref.loc17, element1
// CHECK:STDOUT:   %int.4.loc17: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc17_7.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_7.2: <bound method> = bound_method %int.4.loc17, %.loc17_7.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc17_7.3: <specific function> = specific_function %.loc17_7.2, @Convert.2(constants.%int.32) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %.loc17_7.3(%int.4.loc17) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc17_7.4: init %i32 = converted %int.4.loc17, %int.convert_checked.loc17 [template = constants.%int.4.2]
// CHECK:STDOUT:   assign %.loc17_4, %.loc17_7.4
// CHECK:STDOUT:   %int.32.loc19_15: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_15: init type = call constants.%Int(%int.32.loc19_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.1: type = value_of_initializer %int.make_type_signed.loc19_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.2: type = converted %int.make_type_signed.loc19_15, %.loc19_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int.32.loc19_24: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_24: init type = call constants.%Int(%int.32.loc19_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.1: type = value_of_initializer %int.make_type_signed.loc19_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.2: type = converted %int.make_type_signed.loc19_24, %.loc19_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %c.var: ref %struct_type.a.b.1 = var c
// CHECK:STDOUT:   %c: ref %struct_type.a.b.1 = bind_name c, %c.var
// CHECK:STDOUT:   %int.1.loc19: Core.IntLiteral = int_value 1 [template = constants.%int.1.1]
// CHECK:STDOUT:   %int.2.loc19: Core.IntLiteral = int_value 2 [template = constants.%int.2.1]
// CHECK:STDOUT:   %.loc19_46.1: %struct_type.a.b.2 = struct_literal (%int.1.loc19, %int.2.loc19)
// CHECK:STDOUT:   %.loc19_46.2: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_46.3: <bound method> = bound_method %int.1.loc19, %.loc19_46.2 [template = constants.%.30]
// CHECK:STDOUT:   %.loc19_46.4: <specific function> = specific_function %.loc19_46.3, @Convert.2(constants.%int.32) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.1: init %i32 = call %.loc19_46.4(%int.1.loc19) [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc19_46.5: init %i32 = converted %int.1.loc19, %int.convert_checked.loc19_46.1 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc19_46.6: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc19_46.7: init %i32 = initialize_from %.loc19_46.5 to %.loc19_46.6 [template = constants.%int.1.2]
// CHECK:STDOUT:   %.loc19_46.8: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc19_46.9: <bound method> = bound_method %int.2.loc19, %.loc19_46.8 [template = constants.%.32]
// CHECK:STDOUT:   %.loc19_46.10: <specific function> = specific_function %.loc19_46.9, @Convert.2(constants.%int.32) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.2: init %i32 = call %.loc19_46.10(%int.2.loc19) [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc19_46.11: init %i32 = converted %int.2.loc19, %int.convert_checked.loc19_46.2 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc19_46.12: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc19_46.13: init %i32 = initialize_from %.loc19_46.11 to %.loc19_46.12 [template = constants.%int.2.2]
// CHECK:STDOUT:   %.loc19_46.14: init %struct_type.a.b.1 = struct_init (%.loc19_46.7, %.loc19_46.13) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_47: init %struct_type.a.b.1 = converted %.loc19_46.1, %.loc19_46.14 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc19_47
// CHECK:STDOUT:   %c.ref.loc20: ref %struct_type.a.b.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc20_4: ref %i32 = struct_access %c.ref.loc20, element0
// CHECK:STDOUT:   %int.3.loc20: Core.IntLiteral = int_value 3 [template = constants.%int.3.1]
// CHECK:STDOUT:   %.loc20_7.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc20_7.2: <bound method> = bound_method %int.3.loc20, %.loc20_7.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc20_7.3: <specific function> = specific_function %.loc20_7.2, @Convert.2(constants.%int.32) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %.loc20_7.3(%int.3.loc20) [template = constants.%int.3.2]
// CHECK:STDOUT:   %.loc20_7.4: init %i32 = converted %int.3.loc20, %int.convert_checked.loc20 [template = constants.%int.3.2]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_7.4
// CHECK:STDOUT:   %c.ref.loc21: ref %struct_type.a.b.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc21_4: ref %i32 = struct_access %c.ref.loc21, element1
// CHECK:STDOUT:   %int.4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int.4.1]
// CHECK:STDOUT:   %.loc21_7.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc21_7.2: <bound method> = bound_method %int.4.loc21, %.loc21_7.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc21_7.3: <specific function> = specific_function %.loc21_7.2, @Convert.2(constants.%int.32) [template = constants.%.37]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %.loc21_7.3(%int.4.loc21) [template = constants.%int.4.2]
// CHECK:STDOUT:   %.loc21_7.4: init %i32 = converted %int.4.loc21, %int.convert_checked.loc21 [template = constants.%int.4.2]
// CHECK:STDOUT:   assign %.loc21_4, %.loc21_7.4
// CHECK:STDOUT:   %int.32.loc23: Core.IntLiteral = int_value 32 [template = constants.%int.32]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%int.32.loc23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.1: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.2: type = converted %int.make_type_signed.loc23, %.loc23_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.3: type = ptr_type %i32 [template = constants.%.39]
// CHECK:STDOUT:   %p.var: ref %.39 = var p
// CHECK:STDOUT:   %p: ref %.39 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc23: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc23_17: %.39 = addr_of %a.ref.loc23
// CHECK:STDOUT:   assign %p.var, %.loc23_17
// CHECK:STDOUT:   %p.ref.loc24: ref %.39 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_4: %.39 = bind_value %p.ref.loc24
// CHECK:STDOUT:   %.loc24_3: ref %i32 = deref %.loc24_4
// CHECK:STDOUT:   %int.5: Core.IntLiteral = int_value 5 [template = constants.%int.5.1]
// CHECK:STDOUT:   %.loc24_6.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc24_6.2: <bound method> = bound_method %int.5, %.loc24_6.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc24_6.3: <specific function> = specific_function %.loc24_6.2, @Convert.2(constants.%int.32) [template = constants.%.41]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %.loc24_6.3(%int.5) [template = constants.%int.5.2]
// CHECK:STDOUT:   %.loc24_6.4: init %i32 = converted %int.5, %int.convert_checked.loc24 [template = constants.%int.5.2]
// CHECK:STDOUT:   assign %.loc24_3, %.loc24_6.4
// CHECK:STDOUT:   %.loc26_8: bool = bool_literal true [template = constants.%.42]
// CHECK:STDOUT:   if %.loc26_8 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc26: ref %.39 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_18: %.39 = bind_value %p.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc26: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %.loc26_25: %.39 = addr_of %a.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_25)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_5: %.39 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc26_3: ref %i32 = deref %.loc26_5
// CHECK:STDOUT:   %int.10: Core.IntLiteral = int_value 10 [template = constants.%int.10.1]
// CHECK:STDOUT:   %.loc26_29.1: %Convert.type.2 = interface_witness_access constants.%.24, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc26_29.2: <bound method> = bound_method %int.10, %.loc26_29.1 [template = constants.%.43]
// CHECK:STDOUT:   %.loc26_29.3: <specific function> = specific_function %.loc26_29.2, @Convert.2(constants.%int.32) [template = constants.%.44]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %.loc26_29.3(%int.10) [template = constants.%int.10.2]
// CHECK:STDOUT:   %.loc26_29.4: init %i32 = converted %int.10, %int.convert_checked.loc26 [template = constants.%int.10.2]
// CHECK:STDOUT:   assign %.loc26_3, %.loc26_29.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
