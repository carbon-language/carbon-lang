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
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.6: (i32, i32) = tuple_value (%.4, %.5) [template]
// CHECK:STDOUT:   %.7: type = struct_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.8: type = ptr_type {.a: i32, .b: i32, .c: i32} [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.10: type = struct_type {.c: i32, .b: i32, .a: i32} [template]
// CHECK:STDOUT:   %.11: type = struct_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.12: type = ptr_type {.b: i32, .a: i32, .c: i32} [template]
// CHECK:STDOUT:   %.13: {.b: i32, .a: i32, .c: i32} = struct_value (%.5, %.4, %.9) [template]
// CHECK:STDOUT:   %.14: {.a: i32, .b: i32, .c: i32} = struct_value (%.4, %.5, %.9) [template]
// CHECK:STDOUT:   %.15: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.16: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.17: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.18: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.19: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.20: i32 = int_literal 7 [template]
// CHECK:STDOUT:   %.21: i32 = int_literal 9 [template]
// CHECK:STDOUT:   %.22: type = tuple_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.23: type = ptr_type (i32, i32, i32, i32) [template]
// CHECK:STDOUT:   %.24: (i32, i32, i32, i32) = tuple_value (%.19, %.20, %.4, %.21) [template]
// CHECK:STDOUT:   %.25: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.26: [i32; 1] = tuple_value (%.17) [template]
// CHECK:STDOUT:   %.27: type = struct_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.28: type = ptr_type {.a: i32, .b: i32} [template]
// CHECK:STDOUT:   %.29: {.a: i32, .b: i32} = struct_value (%.9, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.tuple_copy = %tuple_copy, .struct_copy = %struct_copy, .tuple_index = %tuple_index, .struct_access = %struct_access} [template]
// CHECK:STDOUT:   %.loc7_26.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_26.2: type = converted %.loc7_26.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %tuple_copy.var: ref (i32, i32) = var tuple_copy
// CHECK:STDOUT:   %tuple_copy: ref (i32, i32) = bind_name tuple_copy, %tuple_copy.var
// CHECK:STDOUT:   %.loc7_31: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_34: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_35.1: (i32, i32) = tuple_literal (%.loc7_31, %.loc7_34)
// CHECK:STDOUT:   %.loc7_49.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc7_49.2: type = converted %.loc7_49.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc7_35.2: ref i32 = tuple_access %tuple_copy.var, element0
// CHECK:STDOUT:   %.loc7_35.3: init i32 = initialize_from %.loc7_31 to %.loc7_35.2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_35.4: ref i32 = tuple_access %tuple_copy.var, element1
// CHECK:STDOUT:   %.loc7_35.5: init i32 = initialize_from %.loc7_34 to %.loc7_35.4 [template = constants.%.5]
// CHECK:STDOUT:   %.loc7_35.6: init (i32, i32) = tuple_init (%.loc7_35.3, %.loc7_35.5) to %tuple_copy.var [template = constants.%.6]
// CHECK:STDOUT:   %.loc7_35.7: init (i32, i32) = converted %.loc7_35.1, %.loc7_35.6 [template = constants.%.6]
// CHECK:STDOUT:   assign %tuple_copy.var, %.loc7_35.7
// CHECK:STDOUT:   %.loc9_44: type = struct_type {.a: i32, .b: i32, .c: i32} [template = constants.%.7]
// CHECK:STDOUT:   %struct_copy.var: ref {.a: i32, .b: i32, .c: i32} = var struct_copy
// CHECK:STDOUT:   %struct_copy: ref {.a: i32, .b: i32, .c: i32} = bind_name struct_copy, %struct_copy.var
// CHECK:STDOUT:   %.loc9_54: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_62: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_70: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_71.1: {.c: i32, .b: i32, .a: i32} = struct_literal (%.loc9_54, %.loc9_62, %.loc9_70)
// CHECK:STDOUT:   %.loc9_102: type = struct_type {.b: i32, .a: i32, .c: i32} [template = constants.%.11]
// CHECK:STDOUT:   %.loc9_71.2: {.b: i32, .a: i32, .c: i32} = struct_value (%.loc9_62, %.loc9_70, %.loc9_54) [template = constants.%.13]
// CHECK:STDOUT:   %.loc9_71.3: {.b: i32, .a: i32, .c: i32} = converted %.loc9_71.1, %.loc9_71.2 [template = constants.%.13]
// CHECK:STDOUT:   %.loc9_71.4: i32 = struct_access %.loc9_71.3, element1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_71.5: ref i32 = struct_access %struct_copy.var, element1
// CHECK:STDOUT:   %.loc9_71.6: init i32 = initialize_from %.loc9_71.4 to %.loc9_71.5 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_71.7: i32 = struct_access %.loc9_71.3, element0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_71.8: ref i32 = struct_access %struct_copy.var, element0
// CHECK:STDOUT:   %.loc9_71.9: init i32 = initialize_from %.loc9_71.7 to %.loc9_71.8 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_71.10: i32 = struct_access %.loc9_71.3, element2 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_71.11: ref i32 = struct_access %struct_copy.var, element2
// CHECK:STDOUT:   %.loc9_71.12: init i32 = initialize_from %.loc9_71.10 to %.loc9_71.11 [template = constants.%.9]
// CHECK:STDOUT:   %.loc9_71.13: init {.a: i32, .b: i32, .c: i32} = struct_init (%.loc9_71.6, %.loc9_71.9, %.loc9_71.12) to %struct_copy.var [template = constants.%.14]
// CHECK:STDOUT:   %.loc9_71.14: init {.a: i32, .b: i32, .c: i32} = converted %.loc9_71.3, %.loc9_71.13 [template = constants.%.14]
// CHECK:STDOUT:   assign %struct_copy.var, %.loc9_71.14
// CHECK:STDOUT:   %.loc11_24: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_25: type = array_type %.loc11_24, i32 [template = constants.%.15]
// CHECK:STDOUT:   %tuple_index.var: ref [i32; 1] = var tuple_index
// CHECK:STDOUT:   %tuple_index: ref [i32; 1] = bind_name tuple_index, %tuple_index.var
// CHECK:STDOUT:   %.loc11_30: i32 = int_literal 0 [template = constants.%.17]
// CHECK:STDOUT:   %.loc11_32.1: (i32,) = tuple_literal (%.loc11_30)
// CHECK:STDOUT:   %.loc11_44: i32 = int_literal 5 [template = constants.%.19]
// CHECK:STDOUT:   %.loc11_47: i32 = int_literal 7 [template = constants.%.20]
// CHECK:STDOUT:   %.loc11_50: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_53: i32 = int_literal 9 [template = constants.%.21]
// CHECK:STDOUT:   %.loc11_54.1: (i32, i32, i32, i32) = tuple_literal (%.loc11_44, %.loc11_47, %.loc11_50, %.loc11_53)
// CHECK:STDOUT:   %.loc11_56: i32 = int_literal 2 [template = constants.%.5]
// CHECK:STDOUT:   %.loc11_54.2: (i32, i32, i32, i32) = tuple_value (%.loc11_44, %.loc11_47, %.loc11_50, %.loc11_53) [template = constants.%.24]
// CHECK:STDOUT:   %.loc11_54.3: (i32, i32, i32, i32) = converted %.loc11_54.1, %.loc11_54.2 [template = constants.%.24]
// CHECK:STDOUT:   %.loc11_57: i32 = tuple_index %.loc11_54.3, %.loc11_56 [template = constants.%.4]
// CHECK:STDOUT:   %.loc11_58: type = array_type %.loc11_57, i32 [template = constants.%.15]
// CHECK:STDOUT:   %.loc11_5: ref [i32; 1] = splice_block %tuple_index.var {}
// CHECK:STDOUT:   %.loc11_32.2: i32 = int_literal 0 [template = constants.%.25]
// CHECK:STDOUT:   %.loc11_32.3: ref i32 = array_index %.loc11_5, %.loc11_32.2
// CHECK:STDOUT:   %.loc11_32.4: init i32 = initialize_from %.loc11_30 to %.loc11_32.3 [template = constants.%.17]
// CHECK:STDOUT:   %.loc11_32.5: init [i32; 1] = array_init (%.loc11_32.4) to %.loc11_5 [template = constants.%.26]
// CHECK:STDOUT:   %.loc11_32.6: init [i32; 1] = converted %.loc11_32.1, %.loc11_32.5 [template = constants.%.26]
// CHECK:STDOUT:   assign %tuple_index.var, %.loc11_32.6
// CHECK:STDOUT:   %.loc13_26: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_27: type = array_type %.loc13_26, i32 [template = constants.%.15]
// CHECK:STDOUT:   %struct_access.var: ref [i32; 1] = var struct_access
// CHECK:STDOUT:   %struct_access: ref [i32; 1] = bind_name struct_access, %struct_access.var
// CHECK:STDOUT:   %.loc13_32: i32 = int_literal 0 [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_34.1: (i32,) = tuple_literal (%.loc13_32)
// CHECK:STDOUT:   %.loc13_51: i32 = int_literal 3 [template = constants.%.9]
// CHECK:STDOUT:   %.loc13_59: i32 = int_literal 1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_60.1: {.a: i32, .b: i32} = struct_literal (%.loc13_51, %.loc13_59)
// CHECK:STDOUT:   %.loc13_60.2: {.a: i32, .b: i32} = struct_value (%.loc13_51, %.loc13_59) [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_60.3: {.a: i32, .b: i32} = converted %.loc13_60.1, %.loc13_60.2 [template = constants.%.29]
// CHECK:STDOUT:   %.loc13_61: i32 = struct_access %.loc13_60.3, element1 [template = constants.%.4]
// CHECK:STDOUT:   %.loc13_63: type = array_type %.loc13_61, i32 [template = constants.%.15]
// CHECK:STDOUT:   %.loc13_5: ref [i32; 1] = splice_block %struct_access.var {}
// CHECK:STDOUT:   %.loc13_34.2: i32 = int_literal 0 [template = constants.%.25]
// CHECK:STDOUT:   %.loc13_34.3: ref i32 = array_index %.loc13_5, %.loc13_34.2
// CHECK:STDOUT:   %.loc13_34.4: init i32 = initialize_from %.loc13_32 to %.loc13_34.3 [template = constants.%.17]
// CHECK:STDOUT:   %.loc13_34.5: init [i32; 1] = array_init (%.loc13_34.4) to %.loc13_5 [template = constants.%.26]
// CHECK:STDOUT:   %.loc13_34.6: init [i32; 1] = converted %.loc13_34.1, %.loc13_34.5 [template = constants.%.26]
// CHECK:STDOUT:   assign %struct_access.var, %.loc13_34.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
