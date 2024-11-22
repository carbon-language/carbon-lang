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
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%i32, %i32) [template]
// CHECK:STDOUT:   %.3: Core.IntLiteral = int_value 1 [template]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 2 [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.28: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.29: <bound method> = bound_method %.3, %Convert.14 [template]
// CHECK:STDOUT:   %.30: <specific function> = specific_function %.29, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.31: %i32 = int_value 1 [template]
// CHECK:STDOUT:   %.32: <bound method> = bound_method %.4, %Convert.14 [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %.32, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.34: %i32 = int_value 2 [template]
// CHECK:STDOUT:   %tuple.1: %tuple.type.2 = tuple_value (%.31, %.34) [template]
// CHECK:STDOUT:   %.35: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %.37: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.38: type = struct_type {.c: Core.IntLiteral, .b: Core.IntLiteral, .a: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.39: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [template]
// CHECK:STDOUT:   %.41: <bound method> = bound_method %.37, %Convert.14 [template]
// CHECK:STDOUT:   %.42: <specific function> = specific_function %.41, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.43: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct.1: %.39 = struct_value (%.34, %.31, %.43) [template]
// CHECK:STDOUT:   %struct.2: %.35 = struct_value (%.31, %.34, %.43) [template]
// CHECK:STDOUT:   %.44: type = array_type %.3, %i32 [template]
// CHECK:STDOUT:   %.46: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %tuple.type.4: type = tuple_type (Core.IntLiteral) [template]
// CHECK:STDOUT:   %.47: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.48: Core.IntLiteral = int_value 7 [template]
// CHECK:STDOUT:   %.49: Core.IntLiteral = int_value 9 [template]
// CHECK:STDOUT:   %tuple.type.5: type = tuple_type (Core.IntLiteral, Core.IntLiteral, Core.IntLiteral, Core.IntLiteral) [template]
// CHECK:STDOUT:   %tuple.2: %tuple.type.5 = tuple_value (%.47, %.48, %.3, %.49) [template]
// CHECK:STDOUT:   %.51: <bound method> = bound_method %.46, %Convert.14 [template]
// CHECK:STDOUT:   %.52: <specific function> = specific_function %.51, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.53: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %array: %.44 = tuple_value (%.53) [template]
// CHECK:STDOUT:   %.54: type = struct_type {.a: Core.IntLiteral, .b: Core.IntLiteral} [template]
// CHECK:STDOUT:   %struct.3: %.54 = struct_value (%.37, %.3) [template]
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
// CHECK:STDOUT:   %.loc11_18: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_18: init type = call constants.%Int(%.loc11_18) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_23: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_23: init type = call constants.%Int(%.loc11_23) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_18, %int.make_type_signed.loc11_23)
// CHECK:STDOUT:   %.loc11_26.2: type = value_of_initializer %int.make_type_signed.loc11_18 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.3: type = converted %int.make_type_signed.loc11_18, %.loc11_26.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.4: type = value_of_initializer %int.make_type_signed.loc11_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.5: type = converted %int.make_type_signed.loc11_23, %.loc11_26.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_26.6: type = converted %.loc11_26.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %tuple_copy.var: ref %tuple.type.2 = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref %tuple.type.2 = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %.loc13_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_23: init type = call constants.%Int(%.loc13_23.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.2: type = value_of_initializer %int.make_type_signed.loc13_23 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_23.3: type = converted %int.make_type_signed.loc13_23, %.loc13_23.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_32: init type = call constants.%Int(%.loc13_32.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.2: type = value_of_initializer %int.make_type_signed.loc13_32 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.3: type = converted %int.make_type_signed.loc13_32, %.loc13_32.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_41.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_41: init type = call constants.%Int(%.loc13_41.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_41.2: type = value_of_initializer %int.make_type_signed.loc13_41 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_41.3: type = converted %int.make_type_signed.loc13_41, %.loc13_41.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_44: type = struct_type {.a: %i32, .b: %i32, .c: %i32} [template = constants.%.35]
// CHECK:STDOUT:   %struct_copy.var: ref %.35 = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref %.35 = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %.loc15_19.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_19.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_24: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_19.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_19.3: type = converted %int.make_type_signed.loc15, %.loc15_19.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_25: type = array_type %.loc15_24, %i32 [template = constants.%.44]
// CHECK:STDOUT:   %tuple_index.var: ref %.44 = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref %.44 = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %.loc17_21.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_21.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_26: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_21.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_21.3: type = converted %int.make_type_signed.loc17, %.loc17_21.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_27: type = array_type %.loc17_26, %i32 [template = constants.%.44]
// CHECK:STDOUT:   %struct_access.var: ref %.44 = var struct_access
// CHECK:STDOUT:   %struct_access: ref %.44 = bind_name struct_access, %struct_access.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_31: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_34: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_35.1: %tuple.type.3 = tuple_literal (%.loc11_31, %.loc11_34)
// CHECK:STDOUT:   %.loc11_41: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_41: init type = call constants.%Int(%.loc11_41) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_46: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc11_46: init type = call constants.%Int(%.loc11_46) [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.1: %tuple.type.1 = tuple_literal (%int.make_type_signed.loc11_41, %int.make_type_signed.loc11_46)
// CHECK:STDOUT:   %.loc11_49.2: type = value_of_initializer %int.make_type_signed.loc11_41 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.3: type = converted %int.make_type_signed.loc11_41, %.loc11_49.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.4: type = value_of_initializer %int.make_type_signed.loc11_46 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.5: type = converted %int.make_type_signed.loc11_46, %.loc11_49.4 [template = constants.%i32]
// CHECK:STDOUT:   %.loc11_49.6: type = converted %.loc11_49.1, constants.%tuple.type.2 [template = constants.%tuple.type.2]
// CHECK:STDOUT:   %.loc11_35.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_35.3: <bound method> = bound_method %.loc11_31, %.loc11_35.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc11_35.4: <specific function> = specific_function %.loc11_35.3, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.1: init %i32 = call %.loc11_35.4(%.loc11_31) [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_35.5: %i32 = value_of_initializer %int.convert_checked.loc11_35.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_35.6: %i32 = converted %.loc11_31, %.loc11_35.5 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_35.7: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_35.8: <bound method> = bound_method %.loc11_34, %.loc11_35.7 [template = constants.%.32]
// CHECK:STDOUT:   %.loc11_35.9: <specific function> = specific_function %.loc11_35.8, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc11_35.2: init %i32 = call %.loc11_35.9(%.loc11_34) [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_35.10: %i32 = value_of_initializer %int.convert_checked.loc11_35.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_35.11: %i32 = converted %.loc11_34, %.loc11_35.10 [template = constants.%.34]
// CHECK:STDOUT:   %tuple.loc11: %tuple.type.2 = tuple_value (%.loc11_35.6, %.loc11_35.11) [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_37.1: %tuple.type.2 = converted %.loc11_35.1, %tuple.loc11 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_37.2: %i32 = tuple_access %.loc11_37.1, element0 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_37.3: ref %i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc11_37.4: init %i32 = initialize_from %.loc11_37.2 to %.loc11_37.3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc11_37.5: %i32 = tuple_access %.loc11_37.1, element1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_37.6: ref %i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc11_37.7: init %i32 = initialize_from %.loc11_37.5 to %.loc11_37.6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_37.8: init %tuple.type.2 = tuple_init (%.loc11_37.4, %.loc11_37.7) to file.%tuple_copy.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_50: init %tuple.type.2 = converted %.loc11_37.1, %.loc11_37.8 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc11_50
// CHECK:STDOUT:   %.loc13_54: Core.IntLiteral = int_value 3 [template = constants.%.37]
// CHECK:STDOUT:   %.loc13_62: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_70: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc13_71.1: %.38 = struct_literal (%.loc13_54, %.loc13_62, %.loc13_70)
// CHECK:STDOUT:   %.loc13_81.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_81: init type = call constants.%Int(%.loc13_81.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_81.2: type = value_of_initializer %int.make_type_signed.loc13_81 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_81.3: type = converted %int.make_type_signed.loc13_81, %.loc13_81.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_90.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_90: init type = call constants.%Int(%.loc13_90.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_90.2: type = value_of_initializer %int.make_type_signed.loc13_90 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_90.3: type = converted %int.make_type_signed.loc13_90, %.loc13_90.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_99.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc13_99: init type = call constants.%Int(%.loc13_99.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_99.2: type = value_of_initializer %int.make_type_signed.loc13_99 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_99.3: type = converted %int.make_type_signed.loc13_99, %.loc13_99.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc13_102: type = struct_type {.b: %i32, .a: %i32, .c: %i32} [template = constants.%.39]
// CHECK:STDOUT:   %.loc13_71.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_71.3: <bound method> = bound_method %.loc13_62, %.loc13_71.2 [template = constants.%.32]
// CHECK:STDOUT:   %.loc13_71.4: <specific function> = specific_function %.loc13_71.3, @Convert.2(constants.%.1) [template = constants.%.33]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.1: init %i32 = call %.loc13_71.4(%.loc13_62) [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_71.5: %i32 = value_of_initializer %int.convert_checked.loc13_71.1 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_71.6: %i32 = converted %.loc13_62, %.loc13_71.5 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_71.7: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_71.8: <bound method> = bound_method %.loc13_70, %.loc13_71.7 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_71.9: <specific function> = specific_function %.loc13_71.8, @Convert.2(constants.%.1) [template = constants.%.30]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.2: init %i32 = call %.loc13_71.9(%.loc13_70) [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_71.10: %i32 = value_of_initializer %int.convert_checked.loc13_71.2 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_71.11: %i32 = converted %.loc13_70, %.loc13_71.10 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_71.12: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc13_71.13: <bound method> = bound_method %.loc13_54, %.loc13_71.12 [template = constants.%.41]
// CHECK:STDOUT:   %.loc13_71.14: <specific function> = specific_function %.loc13_71.13, @Convert.2(constants.%.1) [template = constants.%.42]
// CHECK:STDOUT:   %int.convert_checked.loc13_71.3: init %i32 = call %.loc13_71.14(%.loc13_54) [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_71.15: %i32 = value_of_initializer %int.convert_checked.loc13_71.3 [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_71.16: %i32 = converted %.loc13_54, %.loc13_71.15 [template = constants.%.43]
// CHECK:STDOUT:   %struct.loc13: %.39 = struct_value (%.loc13_71.6, %.loc13_71.11, %.loc13_71.16) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.1: %.39 = converted %.loc13_71.1, %struct.loc13 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.2: %i32 = struct_access %.loc13_73.1, element1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_73.3: ref %i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc13_73.4: init %i32 = initialize_from %.loc13_73.2 to %.loc13_73.3 [template = constants.%.31]
// CHECK:STDOUT:   %.loc13_73.5: %i32 = struct_access %.loc13_73.1, element0 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_73.6: ref %i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc13_73.7: init %i32 = initialize_from %.loc13_73.5 to %.loc13_73.6 [template = constants.%.34]
// CHECK:STDOUT:   %.loc13_73.8: %i32 = struct_access %.loc13_73.1, element2 [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_73.9: ref %i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc13_73.10: init %i32 = initialize_from %.loc13_73.8 to %.loc13_73.9 [template = constants.%.43]
// CHECK:STDOUT:   %.loc13_73.11: init %.35 = struct_init (%.loc13_73.4, %.loc13_73.7, %.loc13_73.10) to file.%struct_copy.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_103: init %.35 = converted %.loc13_73.1, %.loc13_73.11 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc13_103
// CHECK:STDOUT:   %.loc15_30: Core.IntLiteral = int_value 0 [template = constants.%.46]
// CHECK:STDOUT:   %.loc15_32.1: %tuple.type.4 = tuple_literal (%.loc15_30)
// CHECK:STDOUT:   %.loc15_38.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc15: init type = call constants.%Int(%.loc15_38.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_44: Core.IntLiteral = int_value 5 [template = constants.%.47]
// CHECK:STDOUT:   %.loc15_47: Core.IntLiteral = int_value 7 [template = constants.%.48]
// CHECK:STDOUT:   %.loc15_50: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_53: Core.IntLiteral = int_value 9 [template = constants.%.49]
// CHECK:STDOUT:   %.loc15_54.1: %tuple.type.5 = tuple_literal (%.loc15_44, %.loc15_47, %.loc15_50, %.loc15_53)
// CHECK:STDOUT:   %.loc15_56: Core.IntLiteral = int_value 2 [template = constants.%.4]
// CHECK:STDOUT:   %tuple.loc15: %tuple.type.5 = tuple_value (%.loc15_44, %.loc15_47, %.loc15_50, %.loc15_53) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc15_54.2: %tuple.type.5 = converted %.loc15_54.1, %tuple.loc15 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc15_55: Core.IntLiteral = tuple_access %.loc15_54.2, element2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc15_38.2: type = value_of_initializer %int.make_type_signed.loc15 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_38.3: type = converted %int.make_type_signed.loc15, %.loc15_38.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc15_57: type = array_type %.loc15_55, %i32 [template = constants.%.44]
// CHECK:STDOUT:   %.loc15_32.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_32.3: <bound method> = bound_method %.loc15_30, %.loc15_32.2 [template = constants.%.51]
// CHECK:STDOUT:   %.loc15_32.4: <specific function> = specific_function %.loc15_32.3, @Convert.2(constants.%.1) [template = constants.%.52]
// CHECK:STDOUT:   %int.convert_checked.loc15: init %i32 = call %.loc15_32.4(%.loc15_30) [template = constants.%.53]
// CHECK:STDOUT:   %.loc15_32.5: init %i32 = converted %.loc15_30, %int.convert_checked.loc15 [template = constants.%.53]
// CHECK:STDOUT:   %.loc15_5: ref %.44 = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %.loc15_32.6: Core.IntLiteral = int_value 0 [template = constants.%.46]
// CHECK:STDOUT:   %.loc15_32.7: ref %i32 = array_index %.loc15_5, %.loc15_32.6
// CHECK:STDOUT:   %.loc15_32.8: init %i32 = initialize_from %.loc15_32.5 to %.loc15_32.7 [template = constants.%.53]
// CHECK:STDOUT:   %.loc15_32.9: init %.44 = array_init (%.loc15_32.8) to %.loc15_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc15_34: init %.44 = converted %.loc15_32.1, %.loc15_32.9 [template = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc15_34
// CHECK:STDOUT:   %.loc17_32: Core.IntLiteral = int_value 0 [template = constants.%.46]
// CHECK:STDOUT:   %.loc17_34.1: %tuple.type.4 = tuple_literal (%.loc17_32)
// CHECK:STDOUT:   %.loc17_40.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc17: init type = call constants.%Int(%.loc17_40.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_51: Core.IntLiteral = int_value 3 [template = constants.%.37]
// CHECK:STDOUT:   %.loc17_59: Core.IntLiteral = int_value 1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_60.1: %.54 = struct_literal (%.loc17_51, %.loc17_59)
// CHECK:STDOUT:   %struct.loc17: %.54 = struct_value (%.loc17_51, %.loc17_59) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_60.2: %.54 = converted %.loc17_60.1, %struct.loc17 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_61: Core.IntLiteral = struct_access %.loc17_60.2, element1 [template = constants.%.3]
// CHECK:STDOUT:   %.loc17_40.2: type = value_of_initializer %int.make_type_signed.loc17 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_40.3: type = converted %int.make_type_signed.loc17, %.loc17_40.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc17_63: type = array_type %.loc17_61, %i32 [template = constants.%.44]
// CHECK:STDOUT:   %.loc17_34.2: %Convert.type.2 = interface_witness_access constants.%.28, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc17_34.3: <bound method> = bound_method %.loc17_32, %.loc17_34.2 [template = constants.%.51]
// CHECK:STDOUT:   %.loc17_34.4: <specific function> = specific_function %.loc17_34.3, @Convert.2(constants.%.1) [template = constants.%.52]
// CHECK:STDOUT:   %int.convert_checked.loc17: init %i32 = call %.loc17_34.4(%.loc17_32) [template = constants.%.53]
// CHECK:STDOUT:   %.loc17_34.5: init %i32 = converted %.loc17_32, %int.convert_checked.loc17 [template = constants.%.53]
// CHECK:STDOUT:   %.loc17_5: ref %.44 = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %.loc17_34.6: Core.IntLiteral = int_value 0 [template = constants.%.46]
// CHECK:STDOUT:   %.loc17_34.7: ref %i32 = array_index %.loc17_5, %.loc17_34.6
// CHECK:STDOUT:   %.loc17_34.8: init %i32 = initialize_from %.loc17_34.5 to %.loc17_34.7 [template = constants.%.53]
// CHECK:STDOUT:   %.loc17_34.9: init %.44 = array_init (%.loc17_34.8) to %.loc17_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc17_36: init %.44 = converted %.loc17_34.1, %.loc17_34.9 [template = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc17_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
