// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

var tuple_copy: (i32, i32) = (1, 2) as (i32, i32);

var struct_copy: {.a: i32, .b: i32, .c: i32} = {.c = 3, .b = 2, .a = 1} as {.b: i32, .a: i32, .c: i32};

var tuple_index: [i32; 1] = (0,) as [i32; (5, 7, 1, 9)[2]];

var struct_access: [i32; 1] = (0,) as [i32; {.a = 3, .b = 1}.b];

// CHECK:STDOUT: --- aggregate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.3: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %tuple.1: (i32, i32) = tuple_value (%.5, %.6) [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.c: i32, .b: i32, .a: i32} [template]
// CHECK:STDOUT:   %.11: type = struct_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %struct.2: {.b: i32, .a: i32, .c: i32} = struct_value (%.6, %.5, %.9) [template]
// CHECK:STDOUT:   %struct.3: {.a: i32, .b: i32, .c: i32} = struct_value (%.5, %.6, %.9) [template]
// CHECK:STDOUT:   %.13: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.14: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.16: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.18: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.20: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.21: type = ptr_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %tuple.2: (i32, i32, i32, i32) = tuple_value (%.17, %.18, %.5, %.19) [template]
// CHECK:STDOUT:   %array: [i32; 1] = tuple_value (%.15) [template]
// CHECK:STDOUT:   %.22: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.23: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %struct.4: {.a: i32, .b: i32} = struct_value (%.9, %.5) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .tuple_copy = %tuple_copy
// CHECK:STDOUT:     .struct_copy = %struct_copy
// CHECK:STDOUT:     .tuple_index = %tuple_index
// CHECK:STDOUT:     .struct_access = %struct_access
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc7_18: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc7_23: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc7_26.1: (type, type) = tuple_literal (%int.make_type_32.loc7_18, %int.make_type_32.loc7_23)
// CHECK:STDOUT:   %.loc7_26.2: type = value_of_initializer %int.make_type_32.loc7_18 [template = i32]
// CHECK:STDOUT:   %.loc7_26.3: type = converted %int.make_type_32.loc7_18, %.loc7_26.2 [template = i32]
// CHECK:STDOUT:   %.loc7_26.4: type = value_of_initializer %int.make_type_32.loc7_23 [template = i32]
// CHECK:STDOUT:   %.loc7_26.5: type = converted %int.make_type_32.loc7_23, %.loc7_26.4 [template = i32]
// CHECK:STDOUT:   %.loc7_26.6: type = converted %.loc7_26.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %tuple_copy.var: ref (i32, i32) = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref (i32, i32) = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %import_ref.3: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.4: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.5: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc9_23: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_23.1: type = value_of_initializer %int.make_type_32.loc9_23 [template = i32]
// CHECK:STDOUT:   %.loc9_23.2: type = converted %int.make_type_32.loc9_23, %.loc9_23.1 [template = i32]
// CHECK:STDOUT:   %import_ref.6: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc9_32: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_32.1: type = value_of_initializer %int.make_type_32.loc9_32 [template = i32]
// CHECK:STDOUT:   %.loc9_32.2: type = converted %int.make_type_32.loc9_32, %.loc9_32.1 [template = i32]
// CHECK:STDOUT:   %import_ref.7: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc9_41: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_41.1: type = value_of_initializer %int.make_type_32.loc9_41 [template = i32]
// CHECK:STDOUT:   %.loc9_41.2: type = converted %int.make_type_32.loc9_41, %.loc9_41.1 [template = i32]
// CHECK:STDOUT:   %.loc9_44: type = struct_type {.a: i32, .b: i32, .c: i32} [template = constants.%.7]
// CHECK:STDOUT:   %struct_copy.var: ref {.a: i32, .b: i32, .c: i32} = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref {.a: i32, .b: i32, .c: i32} = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %import_ref.8: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.9: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.10: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.11: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_19.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_19.2: type = converted %int.make_type_32.loc11, %.loc11_19.1 [template = i32]
// CHECK:STDOUT:   %.loc11_25: type = array_type %.loc11_24, i32 [template = constants.%.13]
// CHECK:STDOUT:   %tuple_index.var: ref [i32; 1] = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref [i32; 1] = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %import_ref.12: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.13: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_21.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_21.2: type = converted %int.make_type_32.loc13, %.loc13_21.1 [template = i32]
// CHECK:STDOUT:   %.loc13_27: type = array_type %.loc13_26, i32 [template = constants.%.13]
// CHECK:STDOUT:   %struct_access.var: ref [i32; 1] = var struct_access
// CHECK:STDOUT:   %struct_access: ref [i32; 1] = bind_name struct_access, %struct_access.var
// CHECK:STDOUT:   %import_ref.14: Int32 = import_ref ir4, inst+2, loaded [template = constants.%struct.1]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_31: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_34: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_35.1: (i32, i32) = tuple_literal (%.loc7_31, %.loc7_34)
// CHECK:STDOUT:   %int.make_type_32.loc7_41: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc7_46: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc7_49.1: (type, type) = tuple_literal (%int.make_type_32.loc7_41, %int.make_type_32.loc7_46)
// CHECK:STDOUT:   %.loc7_49.2: type = value_of_initializer %int.make_type_32.loc7_41 [template = i32]
// CHECK:STDOUT:   %.loc7_49.3: type = converted %int.make_type_32.loc7_41, %.loc7_49.2 [template = i32]
// CHECK:STDOUT:   %.loc7_49.4: type = value_of_initializer %int.make_type_32.loc7_46 [template = i32]
// CHECK:STDOUT:   %.loc7_49.5: type = converted %int.make_type_32.loc7_46, %.loc7_49.4 [template = i32]
// CHECK:STDOUT:   %.loc7_49.6: type = converted %.loc7_49.1, constants.%.3 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_35.2: ref i32 = tuple_access file.%tuple_copy.var, element0
// CHECK:STDOUT:   %.loc7_35.3: init i32 = initialize_from %.loc7_31 to %.loc7_35.2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_35.4: ref i32 = tuple_access file.%tuple_copy.var, element1
// CHECK:STDOUT:   %.loc7_35.5: init i32 = initialize_from %.loc7_34 to %.loc7_35.4 [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_35.6: init (i32, i32) = tuple_init (%.loc7_35.3, %.loc7_35.5) to file.%tuple_copy.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc7_50: init (i32, i32) = converted %.loc7_35.1, %.loc7_35.6 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%tuple_copy.var, %.loc7_50
// CHECK:STDOUT:   %.loc9_54: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_62: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_70: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_71: {.c: i32, .b: i32, .a: i32} = struct_literal (%.loc9_54, %.loc9_62, %.loc9_70)
// CHECK:STDOUT:   %int.make_type_32.loc9_81: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_81.1: type = value_of_initializer %int.make_type_32.loc9_81 [template = i32]
// CHECK:STDOUT:   %.loc9_81.2: type = converted %int.make_type_32.loc9_81, %.loc9_81.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc9_90: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_90.1: type = value_of_initializer %int.make_type_32.loc9_90 [template = i32]
// CHECK:STDOUT:   %.loc9_90.2: type = converted %int.make_type_32.loc9_90, %.loc9_90.1 [template = i32]
// CHECK:STDOUT:   %int.make_type_32.loc9_99: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc9_99.1: type = value_of_initializer %int.make_type_32.loc9_99 [template = i32]
// CHECK:STDOUT:   %.loc9_99.2: type = converted %int.make_type_32.loc9_99, %.loc9_99.1 [template = i32]
// CHECK:STDOUT:   %.loc9_102: type = struct_type {.b: i32, .a: i32, .c: i32} [template = constants.%.11]
// CHECK:STDOUT:   %struct.loc9: {.b: i32, .a: i32, .c: i32} = struct_value (%.loc9_62, %.loc9_70, %.loc9_54) [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_73.1: {.b: i32, .a: i32, .c: i32} = converted %.loc9_71, %struct.loc9 [template = constants.%struct.2]
// CHECK:STDOUT:   %.loc9_73.2: i32 = struct_access %.loc9_73.1, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_73.3: ref i32 = struct_access file.%struct_copy.var, element1
// CHECK:STDOUT:   %.loc9_73.4: init i32 = initialize_from %.loc9_73.2 to %.loc9_73.3 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_73.5: i32 = struct_access %.loc9_73.1, element0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_73.6: ref i32 = struct_access file.%struct_copy.var, element0
// CHECK:STDOUT:   %.loc9_73.7: init i32 = initialize_from %.loc9_73.5 to %.loc9_73.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc9_73.8: i32 = struct_access %.loc9_73.1, element2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_73.9: ref i32 = struct_access file.%struct_copy.var, element2
// CHECK:STDOUT:   %.loc9_73.10: init i32 = initialize_from %.loc9_73.8 to %.loc9_73.9 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_73.11: init {.a: i32, .b: i32, .c: i32} = struct_init (%.loc9_73.4, %.loc9_73.7, %.loc9_73.10) to file.%struct_copy.var [template = constants.%struct.3]
// CHECK:STDOUT:   %.loc9_103: init {.a: i32, .b: i32, .c: i32} = converted %.loc9_73.1, %.loc9_73.11 [template = constants.%struct.3]
// CHECK:STDOUT:   assign file.%struct_copy.var, %.loc9_103
// CHECK:STDOUT:   %.loc11_30: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_32.1: (i32,) = tuple_literal (%.loc11_30)
// CHECK:STDOUT:   %int.make_type_32.loc11: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc11_44: i32 = int_literal 5 [template = constants.%.17]
// CHECK:STDOUT:   %.loc11_47: i32 = int_literal 7 [template = constants.%.18]
// CHECK:STDOUT:   %.loc11_50: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_53: i32 = int_literal 9 [template = constants.%.19]
// CHECK:STDOUT:   %.loc11_54.1: (i32, i32, i32, i32) = tuple_literal (%.loc11_44, %.loc11_47, %.loc11_50, %.loc11_53)
// CHECK:STDOUT:   %.loc11_56: i32 = int_literal 2 [template = constants.%.6]
// CHECK:STDOUT:   %tuple: (i32, i32, i32, i32) = tuple_value (%.loc11_44, %.loc11_47, %.loc11_50, %.loc11_53) [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc11_54.2: (i32, i32, i32, i32) = converted %.loc11_54.1, %tuple [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc11_57: i32 = tuple_index %.loc11_54.2, %.loc11_56 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_38.1: type = value_of_initializer %int.make_type_32.loc11 [template = i32]
// CHECK:STDOUT:   %.loc11_38.2: type = converted %int.make_type_32.loc11, %.loc11_38.1 [template = i32]
// CHECK:STDOUT:   %.loc11_58: type = array_type %.loc11_57, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc11_5: ref [i32; 1] = splice_block file.%tuple_index.var {}
// CHECK:STDOUT:   %.loc11_32.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_32.3: ref i32 = array_index %.loc11_5, %.loc11_32.2
// CHECK:STDOUT:   %.loc11_32.4: init i32 = initialize_from %.loc11_30 to %.loc11_32.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_32.5: init [i32; 1] = array_init (%.loc11_32.4) to %.loc11_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc11_34: init [i32; 1] = converted %.loc11_32.1, %.loc11_32.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%tuple_index.var, %.loc11_34
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc13_34.1: (i32,) = tuple_literal (%.loc13_32)
// CHECK:STDOUT:   %int.make_type_32.loc13: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:   %.loc13_51: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_59: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_60.1: {.a: i32, .b: i32} = struct_literal (%.loc13_51, %.loc13_59)
// CHECK:STDOUT:   %struct.loc13: {.a: i32, .b: i32} = struct_value (%.loc13_51, %.loc13_59) [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc13_60.2: {.a: i32, .b: i32} = converted %.loc13_60.1, %struct.loc13 [template = constants.%struct.4]
// CHECK:STDOUT:   %.loc13_61: i32 = struct_access %.loc13_60.2, element1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc13_40.1: type = value_of_initializer %int.make_type_32.loc13 [template = i32]
// CHECK:STDOUT:   %.loc13_40.2: type = converted %int.make_type_32.loc13, %.loc13_40.1 [template = i32]
// CHECK:STDOUT:   %.loc13_63: type = array_type %.loc13_61, i32 [template = constants.%.13]
// CHECK:STDOUT:   %.loc13_5: ref [i32; 1] = splice_block file.%struct_access.var {}
// CHECK:STDOUT:   %.loc13_34.2: i32 = int_literal 0 [template = constants.%.15]
// CHECK:STDOUT:   %.loc13_34.3: ref i32 = array_index %.loc13_5, %.loc13_34.2
// CHECK:STDOUT:   %.loc13_34.4: init i32 = initialize_from %.loc13_32 to %.loc13_34.3 [template = constants.%.15]
// CHECK:STDOUT:   %.loc13_34.5: init [i32; 1] = array_init (%.loc13_34.4) to %.loc13_5 [template = constants.%array]
// CHECK:STDOUT:   %.loc13_36: init [i32; 1] = converted %.loc13_34.1, %.loc13_34.5 [template = constants.%array]
// CHECK:STDOUT:   assign file.%struct_access.var, %.loc13_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
