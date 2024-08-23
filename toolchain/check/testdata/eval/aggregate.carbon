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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple.1: %.3 = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type %.7 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.c: i32, .b: i32, .a: i32} [template]
// CHECK:STDOUT:   %.11: type = struct_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type %.11 [template]
// CHECK:STDOUT:   %struct.1: %.11 = struct_value (%.6, %.5, %.9) [template]
// CHECK:STDOUT:   %struct.2: %.7 = struct_value (%.5, %.6, %.9) [template]
// CHECK:STDOUT:   %.13: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.14: type = ptr_type %.13 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.16: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.20: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.21: type = ptr_type %.20 [template]
// CHECK:STDOUT:   %tuple.2: %.20 = tuple_value (%.17, %.18, %.5, %.19) [template]
// CHECK:STDOUT:   %array: %.13 = tuple_value (%.15) [template]
// CHECK:STDOUT:   %.22: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.23: type = ptr_type %.22 [template]
// CHECK:STDOUT:   %struct.3: %.22 = struct_value (%.9, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:   %int.make_type_32.loc11_18: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_26.1: %.2 = tuple_literal (%int.make_type_32.loc11_18, %int.make_type_32.loc11_23)
// CHECK:STDOUT:   %.loc11_26.2: type = value_of_initializer %int.make_type_32.loc11_18 [template = i32]
// CHECK:STDOUT:   %.loc11_26.3: type = converted %int.make_type_32.loc11_18, %.loc11_26.2 [template = i32]
// CHECK:STDOUT:   %.loc11_26.4: type = value_of_initializer %int.make_type_32.loc11_23 [template = i32]
// CHECK:STDOUT:   %.loc11_26.5: type = converted %int.make_type_32.loc11_23, %.loc11_26.4 [template = i32]
// CHECK:STDOUT:   %.loc11_26.6: type = converted %.loc11_26.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %tuple_copy.var: ref %.3 = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref %.3 = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %int.make_type_32.loc13_23: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_23.1: type = value_of_initializer %int.make_type_32.loc13_23 [template = i32]
// CHECK:STDOUT:   %.loc13_23.2: type = converted %int.make_type_32.loc13_23, %.loc13_23.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc13_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_32.1: type = value_of_initializer %int.make_type_32.loc13_32 [template = i32]
// CHECK:STDOUT:   %.loc13_32.2: type = converted %int.make_type_32.loc13_32, %.loc13_32.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc13_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_41.1: type = value_of_initializer %int.make_type_32.loc13_41 [template = i32]
// CHECK:STDOUT:   %.loc13_41.2: type = converted %int.make_type_32.loc13_41, %.loc13_41.1 [template = i32]
// CHECK:STDOUT:   %.loc13_44: type = struct_type {.a: i32, .b: i32, .c: i32} [template = constants.%.7]
// CHECK:STDOUT:   %struct_copy.var: ref %.7 = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref %.7 = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_24: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_19.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_19.2: type = converted %int.make_type_32.loc15, %.loc15_19.1 [template = i32]
// CHECK:STDOUT:   %.loc15_25: type = array_type %.loc15_24, i32 [template = constants.%.13]
// CHECK:STDOUT:   %tuple_index.var: ref %.13 = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref %.13 = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_26: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_21.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_21.2: type = converted %int.make_type_32.loc17, %.loc17_21.1 [template = i32]
// CHECK:STDOUT:   %.loc17_27: type = array_type %.loc17_26, i32 [template = constants.%.13]
// CHECK:STDOUT:   %struct_access.var: ref %.13 = var struct_access
// CHECK:STDOUT:   %struct_access: ref %.13 = bind_name struct_access, %struct_access.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc11_31: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_34: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_35.1: %.3 = tuple_literal (%.loc11_31, %.loc11_34)
// CHECK:STDOUT:   %int.make_type_32.loc11_41: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc11_46: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc11_49.1: %.2 = tuple_literal (%int.make_type_32.loc11_41, %int.make_type_32.loc11_46)
// CHECK:STDOUT:   %.loc11_49.2: type = value_of_initializer %int.make_type_32.loc11_41 [template = i32]
// CHECK:STDOUT:   %.loc11_49.3: type = converted %int.make_type_32.loc11_41, %.loc11_49.2 [template = i32]
// CHECK:STDOUT:   %.loc11_49.4: type = value_of_initializer %int.make_type_32.loc11_46 [template = i32]
// CHECK:STDOUT:   %.loc11_49.5: type = converted %int.make_type_32.loc11_46, %.loc11_49.4 [template = i32]
// CHECK:STDOUT:   %.loc11_49.6: type = converted %.loc11_49.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc11_35.2: ref i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc11_35.3: init i32 = initialize_from %.loc11_31 to %.loc11_35.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_35.4: ref i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc11_35.5: init i32 = initialize_from %.loc11_34 to %.loc11_35.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc11_35.6: init %.3 = tuple_init (%.loc11_35.3, %.loc11_35.5) to file.%tuple_copy.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc11_50: init %.3 = converted %.loc11_35.1, %.loc11_35.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc11_50
// CHECK:STDOUT:   %.loc13_54: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_62: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_70: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_71: %.10 = struct_literal (%.loc13_54, %.loc13_62, %.loc13_70)
// CHECK:STDOUT:   %int.make_type_32.loc13_81: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_81.1: type = value_of_initializer %int.make_type_32.loc13_81 [template = i32]
// CHECK:STDOUT:   %.loc13_81.2: type = converted %int.make_type_32.loc13_81, %.loc13_81.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc13_90: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_90.1: type = value_of_initializer %int.make_type_32.loc13_90 [template = i32]
// CHECK:STDOUT:   %.loc13_90.2: type = converted %int.make_type_32.loc13_90, %.loc13_90.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc13_99: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc13_99.1: type = value_of_initializer %int.make_type_32.loc13_99 [template = i32]
// CHECK:STDOUT:   %.loc13_99.2: type = converted %int.make_type_32.loc13_99, %.loc13_99.1 [template = i32]
// CHECK:STDOUT:   %.loc13_102: type = struct_type {.b: i32, .a: i32, .c: i32} [template = constants.%.11]
// CHECK:STDOUT:   %struct.loc13: %.11 = struct_value (%.loc13_62, %.loc13_70, %.loc13_54) [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.1: %.11 = converted %.loc13_71, %struct.loc13 [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc13_73.2: i32 = struct_access %.loc13_73.1, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_73.3: ref i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc13_73.4: init i32 = initialize_from %.loc13_73.2 to %.loc13_73.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_73.5: i32 = struct_access %.loc13_73.1, element0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_73.6: ref i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc13_73.7: init i32 = initialize_from %.loc13_73.5 to %.loc13_73.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc13_73.8: i32 = struct_access %.loc13_73.1, element2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_73.9: ref i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc13_73.10: init i32 = initialize_from %.loc13_73.8 to %.loc13_73.9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_73.11: init %.7 = struct_init (%.loc13_73.4, %.loc13_73.7, %.loc13_73.10) to file.%struct_copy.var [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc13_103: init %.7 = converted %.loc13_73.1, %.loc13_73.11 [template = constants.%struct.2]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc13_103
// CHECK:STDOUT:   %.loc15_30: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc15_32.1: %.16 = tuple_literal (%.loc15_30)
// CHECK:STDOUT:   %int.make_type_32.loc15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc15_44: i32 = int_literal 5 [template = constants.%.17]
// CHECK:STDOUT:   %.loc15_47: i32 = int_literal 7 [template = constants.%.18]
// CHECK:STDOUT:   %.loc15_50: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_53: i32 = int_literal 9 [template = constants.%.19]
// CHECK:STDOUT:   %.loc15_54.1: %.20 = tuple_literal (%.loc15_44, %.loc15_47, %.loc15_50, %.loc15_53)
// CHECK:STDOUT:   %.loc15_56: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %tuple: %.20 = tuple_value (%.loc15_44, %.loc15_47, %.loc15_50, %.loc15_53) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc15_54.2: %.20 = converted %.loc15_54.1, %tuple [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc15_55: i32 = tuple_index %.loc15_54.2, %.loc15_56 [template = constants.%.5]
// CHECK:STDOUT:   %.loc15_38.1: type = value_of_initializer %int.make_type_32.loc15 [template = i32]
// CHECK:STDOUT:   %.loc15_38.2: type = converted %int.make_type_32.loc15, %.loc15_38.1 [template = i32]
// CHECK:STDOUT:   %.loc15_57: type = array_type %.loc15_55, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc15_5: ref %.13 = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %.loc15_32.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc15_32.3: ref i32 = array_index %.loc15_5, %.loc15_32.2
// CHECK:STDOUT:   %.loc15_32.4: init i32 = initialize_from %.loc15_30 to %.loc15_32.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc15_32.5: init %.13 = array_init (%.loc15_32.4) to %.loc15_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc15_34: init %.13 = converted %.loc15_32.1, %.loc15_32.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc15_34
// CHECK:STDOUT:   %.loc17_32: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc17_34.1: %.16 = tuple_literal (%.loc17_32)
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_51: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc17_59: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_60.1: %.22 = struct_literal (%.loc17_51, %.loc17_59)
// CHECK:STDOUT:   %struct.loc17: %.22 = struct_value (%.loc17_51, %.loc17_59) [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_60.2: %.22 = converted %.loc17_60.1, %struct.loc17 [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc17_61: i32 = struct_access %.loc17_60.2, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc17_40.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_40.2: type = converted %int.make_type_32.loc17, %.loc17_40.1 [template = i32]
// CHECK:STDOUT:   %.loc17_63: type = array_type %.loc17_61, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc17_5: ref %.13 = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %.loc17_34.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc17_34.3: ref i32 = array_index %.loc17_5, %.loc17_34.2
// CHECK:STDOUT:   %.loc17_34.4: init i32 = initialize_from %.loc17_32 to %.loc17_34.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc17_34.5: init %.13 = array_init (%.loc17_34.4) to %.loc17_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc17_36: init %.13 = converted %.loc17_34.1, %.loc17_34.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc17_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
