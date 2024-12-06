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
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (bool, %i32) [template]
// CHECK:STDOUT:   %true: bool = bool_literal true [template]
// CHECK:STDOUT:   %int_34.1: Core.IntLiteral = int_value 34 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (bool, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.6: type = fn_type @Convert.1, @ImplicitAs(Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_34.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_34.2: %i32 = int_value 34 [template]
// CHECK:STDOUT:   %tuple: %tuple.type.2 = tuple_value (%true, %int_34.2) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.1: type = struct_type {.index: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.1: %struct_type.index.1 = struct_value (%int_1.1) [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.15: type = fn_type @Convert.6, @As(%i32) [template]
// CHECK:STDOUT:   %Convert.type.16: type = fn_type @Convert.7, @impl.5(%int_32) [template]
// CHECK:STDOUT:   %Convert.16: %Convert.type.16 = struct_value () [template]
// CHECK:STDOUT:   %interface.10: <witness> = interface_witness (%Convert.16) [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_0.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.17: type = fn_type @Convert.4, @impl.3(%int_32) [template]
// CHECK:STDOUT:   %Convert.17: %Convert.type.17 = struct_value () [template]
// CHECK:STDOUT:   %interface.11: <witness> = interface_witness (%Convert.17) [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_0.2, %Convert.17 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.4(%int_32) [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_1.1, %Convert.16 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.7(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %struct_type.index.2: type = struct_type {.index: %i32} [template]
// CHECK:STDOUT:   %struct.2: %struct_type.index.2 = struct_value (%int_1.2) [template]
// CHECK:STDOUT:   %Convert.bound.5: <bound method> = bound_method %int_1.2, %Convert.17 [template]
// CHECK:STDOUT:   %Convert.specific_fn.5: <specific function> = specific_function %Convert.bound.5, @Convert.4(%int_32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Bool = %import_ref.1
// CHECK:STDOUT:     .Int = %import_ref.2
// CHECK:STDOUT:     .ImplicitAs = %import_ref.3
// CHECK:STDOUT:     .As = %import_ref.39
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
// CHECK:STDOUT:   %int_32.loc11: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11: init type = call constants.%Int(%int_32.loc11) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_18.1: %tuple.type.1 = tuple_literal (%bool.make_type.loc11, %int.make_type_signed.loc11)
// CHECK:STDOUT:   %.loc11_18.2: type = value_of_initializer %bool.make_type.loc11 [template = bool]
// CHECK:STDOUT:   %.loc11_18.3: type = converted %bool.make_type.loc11, %.loc11_18.2 [template = bool]
// CHECK:STDOUT:   %.loc11_18.4: type = value_of_initializer %int.make_type_signed.loc11 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_18.5: type = converted %int.make_type_signed.loc11, %.loc11_18.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_18.6: type = converted %.loc11_18.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %a.var: ref %tuple.type.2 = var a
// CHECK:STDOUT:   %a: ref %tuple.type.2 = bind_name a, %a.var
// CHECK:STDOUT:   %int_32.loc12: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc12: init type = call constants.%Int(%int_32.loc12) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.1: type = value_of_initializer %int.make_type_signed.loc12 [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_8.2: type = converted %int.make_type_signed.loc12, %.loc12_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %bool.make_type.loc13: init type = call constants.%Bool() [template = bool]
// CHECK:STDOUT:   %.loc13_8.1: type = value_of_initializer %bool.make_type.loc13 [template = bool]
// CHECK:STDOUT:   %.loc13_8.2: type = converted %bool.make_type.loc13, %.loc13_8.1 [template = bool]
// CHECK:STDOUT:   %c.var: ref bool = var c
// CHECK:STDOUT:   %c: ref bool = bind_name c, %c.var
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_8.2: type = converted %int.make_type_signed.loc14, %.loc14_8.1 [template = constants.%i32]
// CHECK:STDOUT:   %d.var: ref %i32 = var d
// CHECK:STDOUT:   %d: ref %i32 = bind_name d, %d.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %true: bool = bool_literal true [template = constants.%true]
// CHECK:STDOUT:   %int_34: Core.IntLiteral = int_value 34 [template = constants.%int_34.1]
// CHECK:STDOUT:   %.loc11_31.1: %tuple.type.3 = tuple_literal (%true, %int_34)
// CHECK:STDOUT:   %tuple.elem0.loc11: ref bool = tuple_access file.%a.var, element0
// CHECK:STDOUT:   %.loc11_31.2: init bool = initialize_from %true to %tuple.elem0.loc11 [template = constants.%true]
// CHECK:STDOUT:   %impl.elem0.loc11: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11: <bound method> = bound_method %int_34, %impl.elem0.loc11 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11: <specific function> = specific_function %Convert.bound.loc11, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11: init %i32 = call %Convert.specific_fn.loc11(%int_34) [template = constants.%int_34.2]
// CHECK:STDOUT:   %.loc11_31.3: init %i32 = converted %int_34, %int.convert_checked.loc11 [template = constants.%int_34.2]
// CHECK:STDOUT:   %tuple.elem1.loc11: ref %i32 = tuple_access file.%a.var, element1
// CHECK:STDOUT:   %.loc11_31.4: init %i32 = initialize_from %.loc11_31.3 to %tuple.elem1.loc11 [template = constants.%int_34.2]
// CHECK:STDOUT:   %.loc11_31.5: init %tuple.type.2 = tuple_init (%.loc11_31.2, %.loc11_31.4) to file.%a.var [template = constants.%tuple]
// CHECK:STDOUT:   %.loc11_32: init %tuple.type.2 = converted %.loc11_31.1, %.loc11_31.5 [template = constants.%tuple]
// CHECK:STDOUT:   assign file.%a.var, %.loc11_32
// CHECK:STDOUT:   %a.ref.loc12: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc12: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc12_28.1: %struct_type.index.1 = struct_literal (%int_1.loc12)
// CHECK:STDOUT:   %struct.loc12: %struct_type.index.1 = struct_value (%int_1.loc12) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_28.2: %struct_type.index.1 = converted %.loc12_28.1, %struct.loc12 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc12_29: Core.IntLiteral = struct_access %.loc12_28.2, element0 [template = constants.%int_1.1]
// CHECK:STDOUT:   %tuple.elem1.loc12: ref %i32 = tuple_access %a.ref.loc12, element1
// CHECK:STDOUT:   %.loc12_15: %i32 = bind_value %tuple.elem1.loc12
// CHECK:STDOUT:   assign file.%b.var, %.loc12_15
// CHECK:STDOUT:   %a.ref.loc13: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %int_32.loc13: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13: init type = call constants.%Int(%int_32.loc13) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.1: type = value_of_initializer %int.make_type_signed.loc13 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.2: type = converted %int.make_type_signed.loc13, %.loc13_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc13_20.1: %Convert.type.15 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc13_20.1: <bound method> = bound_method %int_0, %impl.elem0.loc13_20.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.1: <specific function> = specific_function %Convert.bound.loc13_20.1, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.1: init %i32 = call %Convert.specific_fn.loc13_20.1(%int_0) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_20.1: %i32 = value_of_initializer %int.convert_checked.loc13_20.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc13_20.2: %i32 = converted %int_0, %.loc13_20.1 [template = constants.%int_0.2]
// CHECK:STDOUT:   %impl.elem0.loc13_20.2: %Convert.type.6 = interface_witness_access constants.%interface.11, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %Convert.bound.loc13_20.2: <bound method> = bound_method %.loc13_20.2, %impl.elem0.loc13_20.2 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_20.2: <specific function> = specific_function %Convert.bound.loc13_20.2, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_20.2: init Core.IntLiteral = call %Convert.specific_fn.loc13_20.2(%.loc13_20.2) [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc13_20.3: Core.IntLiteral = value_of_initializer %int.convert_checked.loc13_20.2 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc13_20.4: Core.IntLiteral = converted %.loc13_20.2, %.loc13_20.3 [template = constants.%int_0.1]
// CHECK:STDOUT:   %tuple.elem0.loc13: ref bool = tuple_access %a.ref.loc13, element0
// CHECK:STDOUT:   %.loc13_16: bool = bind_value %tuple.elem0.loc13
// CHECK:STDOUT:   assign file.%c.var, %.loc13_16
// CHECK:STDOUT:   %a.ref.loc14: ref %tuple.type.2 = name_ref a, file.%a
// CHECK:STDOUT:   %int_1.loc14: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_32.loc14: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc14: init type = call constants.%Int(%int_32.loc14) [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_32.1: type = value_of_initializer %int.make_type_signed.loc14 [template = constants.%i32]
// CHECK:STDOUT:   %.loc14_32.2: type = converted %int.make_type_signed.loc14, %.loc14_32.1 [template = constants.%i32]
// CHECK:STDOUT:   %impl.elem0.loc14_29: %Convert.type.15 = interface_witness_access constants.%interface.10, element0 [template = constants.%Convert.16]
// CHECK:STDOUT:   %Convert.bound.loc14_29: <bound method> = bound_method %int_1.loc14, %impl.elem0.loc14_29 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_29: <specific function> = specific_function %Convert.bound.loc14_29, @Convert.7(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc14_29: init %i32 = call %Convert.specific_fn.loc14_29(%int_1.loc14) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_29.1: %i32 = value_of_initializer %int.convert_checked.loc14_29 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_29.2: %i32 = converted %int_1.loc14, %.loc14_29.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.index.2 = struct_literal (%.loc14_29.2)
// CHECK:STDOUT:   %struct.loc14: %struct_type.index.2 = struct_value (%.loc14_29.2) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_35.2: %struct_type.index.2 = converted %.loc14_35.1, %struct.loc14 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc14_36.1: %i32 = struct_access %.loc14_35.2, element0 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc14_36: %Convert.type.6 = interface_witness_access constants.%interface.11, element0 [template = constants.%Convert.17]
// CHECK:STDOUT:   %Convert.bound.loc14_36: <bound method> = bound_method %.loc14_36.1, %impl.elem0.loc14_36 [template = constants.%Convert.bound.5]
// CHECK:STDOUT:   %Convert.specific_fn.loc14_36: <specific function> = specific_function %Convert.bound.loc14_36, @Convert.4(constants.%int_32) [template = constants.%Convert.specific_fn.5]
// CHECK:STDOUT:   %int.convert_checked.loc14_36: init Core.IntLiteral = call %Convert.specific_fn.loc14_36(%.loc14_36.1) [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_36.2: Core.IntLiteral = value_of_initializer %int.convert_checked.loc14_36 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc14_36.3: Core.IntLiteral = converted %.loc14_36.1, %.loc14_36.2 [template = constants.%int_1.1]
// CHECK:STDOUT:   %tuple.elem1.loc14: ref %i32 = tuple_access %a.ref.loc14, element1
// CHECK:STDOUT:   %.loc14_15: %i32 = bind_value %tuple.elem1.loc14
// CHECK:STDOUT:   assign file.%d.var, %.loc14_15
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
