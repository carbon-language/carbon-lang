// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/aggregate.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/aggregate.carbon

var tuple_copy: (i32, i32) = (1, 2) as (i32, i32);

var struct_copy: {.a: i32, .b: i32, .c: i32} = {.c = 3, .b = 2, .a = 1} as {.b: i32, .a: i32, .c: i32};

var tuple_index: [i32; 1] = (0,) as [i32; (5, 7, 1, 9).2];

var struct_access: [i32; 1] = (0,) as [i32; {.a = 3, .b = 1}.b];

// CHECK:STDOUT: --- aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %int_1.1: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %int_2.1: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %interface.9: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %Convert.bound.1: <bound method> = bound_method %int_1.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.1: <specific function> = specific_function %Convert.bound.1, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_1.2: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %Convert.bound.2: <bound method> = bound_method %int_2.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.2: <specific function> = specific_function %Convert.bound.2, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_2.2: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%int_1.2, %int_2.2) [template]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %int_3.1: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %struct_type.c.b.a: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %Convert.bound.3: <bound method> = bound_method %int_3.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.3: <specific function> = specific_function %Convert.bound.3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.2: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct.1: %struct_type.b.a.c = struct_value (%int_2.2, %int_1.2, %int_3.2) [template]
// CHECK:STDOUT:   %struct.2: %struct_type.a.b.c = struct_value (%int_1.2, %int_2.2, %int_3.2) [template]
// CHECK:STDOUT:   %array_type: type = array_type %int_1.1, %i32 [template]
// CHECK:STDOUT:   %int_0.1: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.5 = tuple_value (%int_5, %int_7, %int_1.1, %int_9) [template]
// CHECK:STDOUT:   %Convert.bound.4: <bound method> = bound_method %int_0.1, %Convert.14 [template]
// CHECK:STDOUT:   %Convert.specific_fn.4: <specific function> = specific_function %Convert.bound.4, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_0.2: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_0.2) [template]
// CHECK:STDOUT:   %struct_type.a.b: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.3: %struct_type.a.b = struct_value (%int_3.1, %int_1.1) [template]
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
// CHECK:STDOUT:     .tuple_copy = %tuple_copy
// CHECK:STDOUT:     .struct_copy = %struct_copy
// CHECK:STDOUT:     .tuple_index = %tuple_index
// CHECK:STDOUT:     .struct_access = %struct_access
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %int_32.loc11_18: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_18: init type = call constants.%Int(%int_32.loc11_18) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_23: init type = call constants.%Int(%int_32.loc11_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_18, %int.make_type_signed.loc11_23)
// CHECK:STDOUT:   %.loc11_26.2: type = value_of_initializer %int.make_type_signed.loc11_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.3: type = converted %int.make_type_signed.loc11_18, %.loc11_26.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.4: type = value_of_initializer %int.make_type_signed.loc11_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.5: type = converted %int.make_type_signed.loc11_23, %.loc11_26.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.6: type = converted %.loc11_26.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.2 = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.2 = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %int_32.loc13_23: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_23: init type = call constants.%Int(%int_32.loc13_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.1: type = value_of_initializer %int.make_type_signed.loc13_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.2: type = converted %int.make_type_signed.loc13_23, %.loc13_23.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_32: init type = call constants.%Int(%int_32.loc13_32) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.1: type = value_of_initializer %int.make_type_signed.loc13_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.2: type = converted %int.make_type_signed.loc13_32, %.loc13_32.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_41: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_41: init type = call constants.%Int(%int_32.loc13_41) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_41.1: type = value_of_initializer %int.make_type_signed.loc13_41 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_41.2: type = converted %int.make_type_signed.loc13_41, %.loc13_41.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.a.b.c: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [template = constants.%struct_type.a.b.c]
// CHECK:STDOUT:   %struct_copy.var: ref %struct_type.a.b.c = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref %struct_type.a.b.c = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc15_19.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.2: type = converted %int.make_type_signed.loc15, %.loc15_19.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc15: type = array_type %int_1.loc15, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %tuple_index.var: ref %array_type = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref %array_type = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int_32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc17_21.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_21.2: type = converted %int.make_type_signed.loc17, %.loc17_21.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc17: type = array_type %int_1.loc17, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %struct_access.var: ref %array_type = var struct_access
// CHECK:STDOUT:   %struct_access: ref %array_type = bind_name struct_access, %struct_access.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_1.loc11: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_2.loc11: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.3 = tuple_literal (%int_1.loc11, %int_2.loc11)
// CHECK:STDOUT:   %int_32.loc11_41: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_41: init type = call constants.%Int(%int_32.loc11_41) [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc11_46: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc11_46: init type = call constants.%Int(%int_32.loc11_46) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_41, %int.make_type_signed.loc11_46)
// CHECK:STDOUT:   %.loc11_49.2: type = value_of_initializer %int.make_type_signed.loc11_41 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.3: type = converted %int.make_type_signed.loc11_41, %.loc11_49.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.4: type = value_of_initializer %int.make_type_signed.loc11_46 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.5: type = converted %int.make_type_signed.loc11_46, %.loc11_49.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.6: type = converted %.loc11_49.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %impl.elem0.loc11_35.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_35.1: <bound method> = bound_method %int_1.loc11, %impl.elem0.loc11_35.1 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_35.1: <specific function> = specific_function %Convert.bound.loc11_35.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.1: init %i32 = call %Convert.specific_fn.loc11_35.1(%int_1.loc11) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_35.2: %i32 = value_of_initializer %int.convert_checked.loc11_35.1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc11_35.3: %i32 = converted %int_1.loc11, %.loc11_35.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc11_35.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc11_35.2: <bound method> = bound_method %int_2.loc11, %impl.elem0.loc11_35.2 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc11_35.2: <specific function> = specific_function %Convert.bound.loc11_35.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.2: init %i32 = call %Convert.specific_fn.loc11_35.2(%int_2.loc11) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_35.4: %i32 = value_of_initializer %int.convert_checked.loc11_35.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_35.5: %i32 = converted %int_2.loc11, %.loc11_35.4 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.loc11: %tuple.type.2 = tuple_value (%.loc11_35.3, %.loc11_35.5) [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_37.1: %tuple.type.2 = converted %.loc11_35.1, %tuple.loc11 [template = constants.%tuple.1]
// CHECK:STDOUT:   %tuple.elem0.loc11_37.1: %i32 = tuple_access %.loc11_37.1, element0 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem0.loc11_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc11_37.2: init %i32 = initialize_from %tuple.elem0.loc11_37.1 to %tuple.elem0.loc11_37.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %tuple.elem1.loc11_37.1: %i32 = tuple_access %.loc11_37.1, element1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %tuple.elem1.loc11_37.2: ref %i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc11_37.3: init %i32 = initialize_from %tuple.elem1.loc11_37.1 to %tuple.elem1.loc11_37.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc11_37.4: init %tuple.type.2 = tuple_init (%.loc11_37.2, %.loc11_37.3) to file.%tuple_copy.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_50: init %tuple.type.2 = converted %.loc11_37.1, %.loc11_37.4 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc11_50
// CHECK:STDOUT:   %int_3.loc13: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_2.loc13: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %int_1.loc13: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc13_71.1: %struct_type.c.b.a = struct_literal (%int_3.loc13, %int_2.loc13, %int_1.loc13)
// CHECK:STDOUT:   %int_32.loc13_81: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_81: init type = call constants.%Int(%int_32.loc13_81) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_81.1: type = value_of_initializer %int.make_type_signed.loc13_81 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_81.2: type = converted %int.make_type_signed.loc13_81, %.loc13_81.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_90: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_90: init type = call constants.%Int(%int_32.loc13_90) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_90.1: type = value_of_initializer %int.make_type_signed.loc13_90 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_90.2: type = converted %int.make_type_signed.loc13_90, %.loc13_90.1 [template = constants.%i32]
// CHECK:STDOUT:   %int_32.loc13_99: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc13_99: init type = call constants.%Int(%int_32.loc13_99) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_99.1: type = value_of_initializer %int.make_type_signed.loc13_99 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_99.2: type = converted %int.make_type_signed.loc13_99, %.loc13_99.1 [template = constants.%i32]
// CHECK:STDOUT:   %struct_type.b.a.c: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [template = constants.%struct_type.b.a.c]
// CHECK:STDOUT:   %impl.elem0.loc13_71.1: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_71.1: <bound method> = bound_method %int_2.loc13, %impl.elem0.loc13_71.1 [template = constants.%Convert.bound.2]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_71.1: <specific function> = specific_function %Convert.bound.loc13_71.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.2]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.1: init %i32 = call %Convert.specific_fn.loc13_71.1(%int_2.loc13) [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_71.2: %i32 = value_of_initializer %int.convert_checked.loc13_71.1 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_71.3: %i32 = converted %int_2.loc13, %.loc13_71.2 [template = constants.%int_2.2]
// CHECK:STDOUT:   %impl.elem0.loc13_71.2: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_71.2: <bound method> = bound_method %int_1.loc13, %impl.elem0.loc13_71.2 [template = constants.%Convert.bound.1]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_71.2: <specific function> = specific_function %Convert.bound.loc13_71.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.1]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.2: init %i32 = call %Convert.specific_fn.loc13_71.2(%int_1.loc13) [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_71.4: %i32 = value_of_initializer %int.convert_checked.loc13_71.2 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_71.5: %i32 = converted %int_1.loc13, %.loc13_71.4 [template = constants.%int_1.2]
// CHECK:STDOUT:   %impl.elem0.loc13_71.3: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc13_71.3: <bound method> = bound_method %int_3.loc13, %impl.elem0.loc13_71.3 [template = constants.%Convert.bound.3]
// CHECK:STDOUT:   %Convert.specific_fn.loc13_71.3: <specific function> = specific_function %Convert.bound.loc13_71.3, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.3]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.3: init %i32 = call %Convert.specific_fn.loc13_71.3(%int_3.loc13) [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_71.6: %i32 = value_of_initializer %int.convert_checked.loc13_71.3 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_71.7: %i32 = converted %int_3.loc13, %.loc13_71.6 [template = constants.%int_3.2]
// CHECK:STDOUT:   %struct.loc13: %struct_type.b.a.c = struct_value (%.loc13_71.3, %.loc13_71.5, %.loc13_71.7) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.1: %struct_type.b.a.c = converted %.loc13_71.1, %struct.loc13 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.2: %i32 = struct_access %.loc13_73.1, element1 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_73.3: ref %i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc13_73.4: init %i32 = initialize_from %.loc13_73.2 to %.loc13_73.3 [template = constants.%int_1.2]
// CHECK:STDOUT:   %.loc13_73.5: %i32 = struct_access %.loc13_73.1, element0 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_73.6: ref %i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc13_73.7: init %i32 = initialize_from %.loc13_73.5 to %.loc13_73.6 [template = constants.%int_2.2]
// CHECK:STDOUT:   %.loc13_73.8: %i32 = struct_access %.loc13_73.1, element2 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_73.9: ref %i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc13_73.10: init %i32 = initialize_from %.loc13_73.8 to %.loc13_73.9 [template = constants.%int_3.2]
// CHECK:STDOUT:   %.loc13_73.11: init %struct_type.a.b.c = struct_init (%.loc13_73.4, %.loc13_73.7, %.loc13_73.10) to file.%struct_copy.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_103: init %struct_type.a.b.c = converted %.loc13_73.1, %.loc13_73.11 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc13_103
// CHECK:STDOUT:   %int_0.loc15_30: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc15_32.1: %tuple.type.4 = tuple_literal (%int_0.loc15_30)
// CHECK:STDOUT:   %int_32.loc15: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%int_32.loc15) [template = constants.%i32]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5]
// CHECK:STDOUT:   %int_7: Core.IntLiteral = int_value 7 [template = constants.%int_7]
// CHECK:STDOUT:   %int_1.loc15: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %int_9: Core.IntLiteral = int_value 9 [template = constants.%int_9]
// CHECK:STDOUT:   %.loc15_54.1: %tuple.type.5 = tuple_literal (%int_5, %int_7, %int_1.loc15, %int_9)
// CHECK:STDOUT:   %int_2.loc15: Core.IntLiteral = int_value 2 [template = constants.%int_2.1]
// CHECK:STDOUT:   %tuple.loc15: %tuple.type.5 = tuple_value (%int_5, %int_7, %int_1.loc15, %int_9) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc15_54.2: %tuple.type.5 = converted %.loc15_54.1, %tuple.loc15 [template = constants.%tuple.2]
// CHECK:STDOUT:   %tuple.elem2: Core.IntLiteral = tuple_access %.loc15_54.2, element2 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc15_38.1: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_38.2: type = converted %int.make_type_signed.loc15, %.loc15_38.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc15: type = array_type %tuple.elem2, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc15: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc15: <bound method> = bound_method %int_0.loc15_30, %impl.elem0.loc15 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc15: <specific function> = specific_function %Convert.bound.loc15, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %Convert.specific_fn.loc15(%int_0.loc15_30) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_32.2: init %i32 = converted %int_0.loc15_30, %int.convert_checked.loc15 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_5: ref %array_type = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %int_0.loc15_32: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc15_32.3: ref %i32 = array_index %.loc15_5, %int_0.loc15_32
// CHECK:STDOUT:   %.loc15_32.4: init %i32 = initialize_from %.loc15_32.2 to %.loc15_32.3 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc15_32.5: init %array_type = array_init (%.loc15_32.4) to %.loc15_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc15_34: init %array_type = converted %.loc15_32.1, %.loc15_32.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc15_34
// CHECK:STDOUT:   %int_0.loc17_32: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc17_34.1: %tuple.type.4 = tuple_literal (%int_0.loc17_32)
// CHECK:STDOUT:   %int_32.loc17: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%int_32.loc17) [template = constants.%i32]
// CHECK:STDOUT:   %int_3.loc17: Core.IntLiteral = int_value 3 [template = constants.%int_3.1]
// CHECK:STDOUT:   %int_1.loc17: Core.IntLiteral = int_value 1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc17_60.1: %struct_type.a.b = struct_literal (%int_3.loc17, %int_1.loc17)
// CHECK:STDOUT:   %struct.loc17: %struct_type.a.b = struct_value (%int_3.loc17, %int_1.loc17) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_60.2: %struct_type.a.b = converted %.loc17_60.1, %struct.loc17 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_61: Core.IntLiteral = struct_access %.loc17_60.2, element1 [template = constants.%int_1.1]
// CHECK:STDOUT:   %.loc17_40.1: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_40.2: type = converted %int.make_type_signed.loc17, %.loc17_40.1 [template = constants.%i32]
// CHECK:STDOUT:   %array_type.loc17: type = array_type %.loc17_61, %i32 [template = constants.%array_type]
// CHECK:STDOUT:   %impl.elem0.loc17: %Convert.type.2 = interface_witness_access constants.%interface.9, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %Convert.bound.loc17: <bound method> = bound_method %int_0.loc17_32, %impl.elem0.loc17 [template = constants.%Convert.bound.4]
// CHECK:STDOUT:   %Convert.specific_fn.loc17: <specific function> = specific_function %Convert.bound.loc17, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.4]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %Convert.specific_fn.loc17(%int_0.loc17_32) [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_34.2: init %i32 = converted %int_0.loc17_32, %int.convert_checked.loc17 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_5: ref %array_type = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %int_0.loc17_34: Core.IntLiteral = int_value 0 [template = constants.%int_0.1]
// CHECK:STDOUT:   %.loc17_34.3: ref %i32 = array_index %.loc17_5, %int_0.loc17_34
// CHECK:STDOUT:   %.loc17_34.4: init %i32 = initialize_from %.loc17_34.2 to %.loc17_34.3 [template = constants.%int_0.2]
// CHECK:STDOUT:   %.loc17_34.5: init %array_type = array_init (%.loc17_34.4) to %.loc17_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc17_36: init %array_type = converted %.loc17_34.1, %.loc17_34.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc17_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
