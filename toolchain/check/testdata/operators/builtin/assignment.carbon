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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %int_12.1: Core.IntLiteral = int_value 12 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_12.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_12.2: %i32 = int_value 12 [template]
// CHECK:STDOUT:   %int_9.1: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_9.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_9.2: %i32 = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %int_4.1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.6: <bound method> = bound_method %int_4.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.6: <specific function> = specific_function %Convert.bound.6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.2: %i32 = int_value 4 [template]
// CHECK:STDOUT:   %struct_type.a.b.1: type = struct_type {.a: %i32, .b: %i32} [template]
// CHECK:STDOUT:   %struct_type.a.b.2: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct: %struct_type.a.b.1 = struct_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %ptr.3: type = ptr_type %i32 [template]
// CHECK:STDOUT:   %int_5.1: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %Convert.bound.7: <bound method> = bound_method %int_5.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.7: <specific function> = specific_function %Convert.bound.7, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.2: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_10.1: Core.IntLiteral = int_value 10 [template]
// CHECK:STDOUT:   %Convert.bound.8: <bound method> = bound_method %int_10.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.8: <specific function> = specific_function %Convert.bound.8, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_10.2: %i32 = int_value 10 [template]
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
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = converted %int.make_type_signed.loc12, %.loc12_10.1 [template = constants.%i32]
// CHECK:STDOUT:   %a.var: ref %i32 = var a
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %a.var
// CHECK:STDOUT:   %int_12: Core.IntLiteral = int_value 12 [template = constants.%int_12.1]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_12, %impl.elem0.loc12 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_12) [template = constants.%int_12.2]
// CHECK:STDOUT:   %.loc12_18: init %i32 = converted %int_12, %int.convert_checked.loc12 [template = constants.%int_12.2]
// CHECK:STDOUT:   assign %a.var, %.loc12_18
// CHECK:STDOUT:   %a.ref.loc13: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9.1]
// CHECK:STDOUT:   %impl.elem0.loc13: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13: <bound method> = bound_method %int_9, %impl.elem0.loc13 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13: <specific function> = specific_function %Convert.bound.loc13, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13: init %i32 = call %Convert.specific_fn.loc13(%int_9) [template = constants.%int_9.2]
// CHECK:STDOUT:   %.loc13: init %i32 = converted %int_9, %int.convert_checked.loc13 [template = constants.%int_9.2]
// CHECK:STDOUT:   assign %a.ref.loc13, %.loc13
// CHECK:STDOUT:   %int_32.loc15_11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_11: init type = call constants.%Int(%int_32.loc15_11) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc15_16: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15_16: init type = call constants.%Int(%int_32.loc15_16) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc15_11, %int.make_type_signed.loc15_16)
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_signed.loc15_11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_signed.loc15_11, %.loc15_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.4: type = value_of_initializer %int.make_type_signed.loc15_16 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.5: type = converted %int.make_type_signed.loc15_16, %.loc15_19.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.6: type = converted %.loc15_19.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %b.var: ref %tuple.type.2 = var b
// CHECK:STDOUT:   %b: ref %tuple.type.2 = bind_name b, %b.var
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc15_28.1: %tuple.type.3 = tuple_literal (%int_1.loc15, %int_2.loc15)
// CHECK:STDOUT:   %impl.elem0.loc15_28.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc15_28.1: <bound method> = bound_method %int_1.loc15, %impl.elem0.loc15_28.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_28.1: <specific function> = specific_function %Convert.bound.loc15_28.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.1: init %i32 = call %Convert.specific_fn.loc15_28.1(%int_1.loc15) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc15_28.2: init %i32 = converted %int_1.loc15, %int.convert_checked.loc15_28.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0.loc15: ref %i32 = tuple_access %b.var, element0
// CHECK:STDOUT:   %.loc15_28.3: init %i32 = initialize_from %.loc15_28.2 to %tuple.elem0.loc15 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc15_28.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc15_28.2: <bound method> = bound_method %int_2.loc15, %impl.elem0.loc15_28.2 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_28.2: <specific function> = specific_function %Convert.bound.loc15_28.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc15_28.2: init %i32 = call %Convert.specific_fn.loc15_28.2(%int_2.loc15) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc15_28.4: init %i32 = converted %int_2.loc15, %int.convert_checked.loc15_28.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem1.loc15: ref %i32 = tuple_access %b.var, element1
// CHECK:STDOUT:   %.loc15_28.5: init %i32 = initialize_from %.loc15_28.4 to %tuple.elem1.loc15 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc15_28.6: init %tuple.type.2 = tuple_init (%.loc15_28.3, %.loc15_28.5) to %b.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc15_29: init %tuple.type.2 = converted %.loc15_28.1, %.loc15_28.6 [template = constants.%tuple]
// CHECK:STDOUT:   assign %b.var, %.loc15_29
// CHECK:STDOUT:   %b.ref.loc16: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0]
// CHECK:STDOUT:   %tuple.elem0.loc16: ref %i32 = tuple_access %b.ref.loc16, element0
// CHECK:STDOUT:   %int_3.loc16: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc16: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc16: <bound method> = bound_method %int_3.loc16, %impl.elem0.loc16 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc16: <specific function> = specific_function %Convert.bound.loc16, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc16: init %i32 = call %Convert.specific_fn.loc16(%int_3.loc16) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc16: init %i32 = converted %int_3.loc16, %int.convert_checked.loc16 [template = constants.%int_3.2]
// CHECK:STDOUT:   assign %tuple.elem0.loc16, %.loc16
// CHECK:STDOUT:   %b.ref.loc17: ref %tuple.type.2 = name_ref b, %b
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %tuple.elem1.loc17: ref %i32 = tuple_access %b.ref.loc17, element1
// CHECK:STDOUT:   %int_4.loc17: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_4.loc17, %impl.elem0.loc17 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int_4.loc17) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc17: init %i32 = converted %int_4.loc17, %int.convert_checked.loc17 [template = constants.%int_4.2]
// CHECK:STDOUT:   assign %tuple.elem1.loc17, %.loc17
// CHECK:STDOUT:   %int_32.loc19_15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_15: init type = call constants.%Int(%int_32.loc19_15) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.1: type = value_of_initializer %int.make_type_signed.loc19_15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_15.2: type = converted %int.make_type_signed.loc19_15, %.loc19_15.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc19_24: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc19_24: init type = call constants.%Int(%int_32.loc19_24) [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.1: type = value_of_initializer %int.make_type_signed.loc19_24 [template = constants.%i32]
// CHECK:STDOUT:   %.loc19_24.2: type = converted %int.make_type_signed.loc19_24, %.loc19_24.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: %i32, .b: %i32} [template = constants.%struct_type.a.b.1]
// CHECK:STDOUT:   %c.var: ref %struct_type.a.b.1 = var c
// CHECK:STDOUT:   %c: ref %struct_type.a.b.1 = bind_name c, %c.var
// CHECK:STDOUT:   %int_1.loc19: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc19: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc19_46.1: %struct_type.a.b.2 = struct_literal (%int_1.loc19, %int_2.loc19)
// CHECK:STDOUT:   %impl.elem0.loc19_46.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc19_46.1: <bound method> = bound_method %int_1.loc19, %impl.elem0.loc19_46.1 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_46.1: <specific function> = specific_function %Convert.bound.loc19_46.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.1: init %i32 = call %Convert.specific_fn.loc19_46.1(%int_1.loc19) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc19_46.2: init %i32 = converted %int_1.loc19, %int.convert_checked.loc19_46.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc19_46.3: ref %i32 = struct_access %c.var, element0
// CHECK:STDOUT:   %.loc19_46.4: init %i32 = initialize_from %.loc19_46.2 to %.loc19_46.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc19_46.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc19_46.2: <bound method> = bound_method %int_2.loc19, %impl.elem0.loc19_46.2 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc19_46.2: <specific function> = specific_function %Convert.bound.loc19_46.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc19_46.2: init %i32 = call %Convert.specific_fn.loc19_46.2(%int_2.loc19) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_46.5: init %i32 = converted %int_2.loc19, %int.convert_checked.loc19_46.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_46.6: ref %i32 = struct_access %c.var, element1
// CHECK:STDOUT:   %.loc19_46.7: init %i32 = initialize_from %.loc19_46.5 to %.loc19_46.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc19_46.8: init %struct_type.a.b.1 = struct_init (%.loc19_46.4, %.loc19_46.7) to %c.var [template = constants.%struct]
// CHECK:STDOUT:   %.loc19_47: init %struct_type.a.b.1 = converted %.loc19_46.1, %.loc19_46.8 [template = constants.%struct]
// CHECK:STDOUT:   assign %c.var, %.loc19_47
// CHECK:STDOUT:   %c.ref.loc20: ref %struct_type.a.b.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc20_4: ref %i32 = struct_access %c.ref.loc20, element0
// CHECK:STDOUT:   %int_3.loc20: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %impl.elem0.loc20: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc20: <bound method> = bound_method %int_3.loc20, %impl.elem0.loc20 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc20: <specific function> = specific_function %Convert.bound.loc20, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc20: init %i32 = call %Convert.specific_fn.loc20(%int_3.loc20) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc20_7: init %i32 = converted %int_3.loc20, %int.convert_checked.loc20 [template = constants.%int_3.2]
// CHECK:STDOUT:   assign %.loc20_4, %.loc20_7
// CHECK:STDOUT:   %c.ref.loc21: ref %struct_type.a.b.1 = name_ref c, %c
// CHECK:STDOUT:   %.loc21_4: ref %i32 = struct_access %c.ref.loc21, element1
// CHECK:STDOUT:   %int_4.loc21: Core.IntLiteral = int_value 4 [template = constants.%int_4.1]
// CHECK:STDOUT:   %impl.elem0.loc21: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc21: <bound method> = bound_method %int_4.loc21, %impl.elem0.loc21 [template = constants.%Convert.bound.6]
// CHECK:STDOUT:   %Convert.specific_fn.loc21: <specific function> = specific_function %Convert.bound.loc21, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.6]
// CHECK:STDOUT:   %int.convert_checked.loc21: init %i32 = call %Convert.specific_fn.loc21(%int_4.loc21) [template = constants.%int_4.2]
// CHECK:STDOUT:   %.loc21_7: init %i32 = converted %int_4.loc21, %int.convert_checked.loc21 [template = constants.%int_4.2]
// CHECK:STDOUT:   assign %.loc21_4, %.loc21_7
// CHECK:STDOUT:   %int_32.loc23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc23: init type = call constants.%Int(%int_32.loc23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.1: type = value_of_initializer %int.make_type_signed.loc23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc23_13.2: type = converted %int.make_type_signed.loc23, %.loc23_13.1 [template = constants.%i32]
// CHECK:STDOUT:   %ptr: type = ptr_type %i32 [template = constants.%ptr.3]
// CHECK:STDOUT:   %p.var: ref %ptr.3 = var p
// CHECK:STDOUT:   %p: ref %ptr.3 = bind_name p, %p.var
// CHECK:STDOUT:   %a.ref.loc23: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc23: %ptr.3 = addr_of %a.ref.loc23
// CHECK:STDOUT:   assign %p.var, %addr.loc23
// CHECK:STDOUT:   %p.ref.loc24: ref %ptr.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc24_4: %ptr.3 = bind_value %p.ref.loc24
// CHECK:STDOUT:   %.loc24_3: ref %i32 = deref %.loc24_4
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.1]
// CHECK:STDOUT:   %impl.elem0.loc24: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc24: <bound method> = bound_method %int_5, %impl.elem0.loc24 [template = constants.%Convert.bound.7]
// CHECK:STDOUT:   %Convert.specific_fn.loc24: <specific function> = specific_function %Convert.bound.loc24, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.7]
// CHECK:STDOUT:   %int.convert_checked.loc24: init %i32 = call %Convert.specific_fn.loc24(%int_5) [template = constants.%int_5.2]
// CHECK:STDOUT:   %.loc24_6: init %i32 = converted %int_5, %int.convert_checked.loc24 [template = constants.%int_5.2]
// CHECK:STDOUT:   assign %.loc24_3, %.loc24_6
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %p.ref.loc26: ref %ptr.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc26_18: %ptr.3 = bind_value %p.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%.loc26_18)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %a.ref.loc26: ref %i32 = name_ref a, %a
// CHECK:STDOUT:   %addr.loc26: %ptr.3 = addr_of %a.ref.loc26
// CHECK:STDOUT:   br !if.expr.result(%addr.loc26)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc26_5: %ptr.3 = block_arg !if.expr.result
// CHECK:STDOUT:   %.loc26_3: ref %i32 = deref %.loc26_5
// CHECK:STDOUT:   %int_10: Core.IntLiteral = int_value 10 [template = constants.%int_10.1]
// CHECK:STDOUT:   %impl.elem0.loc26: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc26: <bound method> = bound_method %int_10, %impl.elem0.loc26 [template = constants.%Convert.bound.8]
// CHECK:STDOUT:   %Convert.specific_fn.loc26: <specific function> = specific_function %Convert.bound.loc26, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.8]
// CHECK:STDOUT:   %int.convert_checked.loc26: init %i32 = call %Convert.specific_fn.loc26(%int_10) [template = constants.%int_10.2]
// CHECK:STDOUT:   %.loc26_29: init %i32 = converted %int_10, %int.convert_checked.loc26 [template = constants.%int_10.2]
// CHECK:STDOUT:   assign %.loc26_3, %.loc26_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
