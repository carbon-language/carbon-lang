// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- implicit.carbon

package Implicit api;

var a_ref: (i32,) = (0,);
var b_ref: (((i32,), i32), (i32, i32)) = (((0,), 1), (2, 3));

// --- implicit.impl.carbon

package Implicit impl;

var a: (i32,) = a_ref;
var b: (((i32,), i32), (i32, i32)) = b_ref;

// CHECK:STDOUT: --- implicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %tuple.1: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %.4: type = tuple_type ((type,), type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.6: type = tuple_type (((type,), type), (type, type)) [template]
// CHECK:STDOUT:   %.7: type = tuple_type ((i32,), i32) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.9: type = tuple_type (((i32,), i32), (i32, i32)) [template]
// CHECK:STDOUT:   %.10: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.11: type = ptr_type ((i32,), i32) [template]
// CHECK:STDOUT:   %.12: type = tuple_type (((i32,), i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.13: type = ptr_type (((i32,), i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.14: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.15: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.16: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %tuple.2: ((i32,), i32) = tuple_value (%tuple.1, %.14) [template]
// CHECK:STDOUT:   %tuple.3: (i32, i32) = tuple_value (%.15, %.16) [template]
// CHECK:STDOUT:   %tuple.4: (((i32,), i32), (i32, i32)) = tuple_value (%tuple.2, %tuple.3) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a_ref = %a_ref
// CHECK:STDOUT:     .b_ref = %b_ref
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc4_17.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc4_17.2: type = converted %.loc4_17.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a_ref.var: ref (i32,) = var a_ref
// CHECK:STDOUT:   %a_ref: ref (i32,) = bind_name a_ref, %a_ref.var
// CHECK:STDOUT:   %.loc5_19: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc5_25: ((type,), type) = tuple_literal (%.loc5_19, i32)
// CHECK:STDOUT:   %.loc5_37: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc5_38.1: (((type,), type), (type, type)) = tuple_literal (%.loc5_25, %.loc5_37)
// CHECK:STDOUT:   %.loc5_38.2: type = converted %.loc5_19, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_38.3: type = converted %.loc5_25, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_38.4: type = converted %.loc5_37, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %.loc5_38.5: type = converted %.loc5_38.1, constants.%.9 [template = constants.%.9]
// CHECK:STDOUT:   %b_ref.var: ref (((i32,), i32), (i32, i32)) = var b_ref
// CHECK:STDOUT:   %b_ref: ref (((i32,), i32), (i32, i32)) = bind_name b_ref, %b_ref.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_22: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_24.1: (i32,) = tuple_literal (%.loc4_22)
// CHECK:STDOUT:   %.loc4_24.2: init (i32,) = tuple_init (%.loc4_22) to file.%a_ref.var [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc4_25: init (i32,) = converted %.loc4_24.1, %.loc4_24.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   assign file.%a_ref.var, %.loc4_25
// CHECK:STDOUT:   %.loc5_45: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc5_47.1: (i32,) = tuple_literal (%.loc5_45)
// CHECK:STDOUT:   %.loc5_50: i32 = int_literal 1 [template = constants.%.14]
// CHECK:STDOUT:   %.loc5_51.1: ((i32,), i32) = tuple_literal (%.loc5_47.1, %.loc5_50)
// CHECK:STDOUT:   %.loc5_55: i32 = int_literal 2 [template = constants.%.15]
// CHECK:STDOUT:   %.loc5_58: i32 = int_literal 3 [template = constants.%.16]
// CHECK:STDOUT:   %.loc5_59.1: (i32, i32) = tuple_literal (%.loc5_55, %.loc5_58)
// CHECK:STDOUT:   %.loc5_60.1: (((i32,), i32), (i32, i32)) = tuple_literal (%.loc5_51.1, %.loc5_59.1)
// CHECK:STDOUT:   %.loc5_60.2: ref ((i32,), i32) = tuple_access file.%b_ref.var, element0
// CHECK:STDOUT:   %.loc5_51.2: ref (i32,) = tuple_access %.loc5_60.2, element0
// CHECK:STDOUT:   %.loc5_47.2: init (i32,) = tuple_init (%.loc5_45) to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.3: init (i32,) = converted %.loc5_47.1, %.loc5_47.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.4: init (i32,) = initialize_from %.loc5_51.3 to %.loc5_51.2 [template = constants.%tuple.1]
// CHECK:STDOUT:   %.loc5_51.5: ref i32 = tuple_access %.loc5_60.2, element1
// CHECK:STDOUT:   %.loc5_51.6: init i32 = initialize_from %.loc5_50 to %.loc5_51.5 [template = constants.%.14]
// CHECK:STDOUT:   %.loc5_51.7: init ((i32,), i32) = tuple_init (%.loc5_51.4, %.loc5_51.6) to %.loc5_60.2 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.3: init ((i32,), i32) = converted %.loc5_51.1, %.loc5_51.7 [template = constants.%tuple.2]
// CHECK:STDOUT:   %.loc5_60.4: ref (i32, i32) = tuple_access file.%b_ref.var, element1
// CHECK:STDOUT:   %.loc5_59.2: ref i32 = tuple_access %.loc5_60.4, element0
// CHECK:STDOUT:   %.loc5_59.3: init i32 = initialize_from %.loc5_55 to %.loc5_59.2 [template = constants.%.15]
// CHECK:STDOUT:   %.loc5_59.4: ref i32 = tuple_access %.loc5_60.4, element1
// CHECK:STDOUT:   %.loc5_59.5: init i32 = initialize_from %.loc5_58 to %.loc5_59.4 [template = constants.%.16]
// CHECK:STDOUT:   %.loc5_59.6: init (i32, i32) = tuple_init (%.loc5_59.3, %.loc5_59.5) to %.loc5_60.4 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.5: init (i32, i32) = converted %.loc5_59.1, %.loc5_59.6 [template = constants.%tuple.3]
// CHECK:STDOUT:   %.loc5_60.6: init (((i32,), i32), (i32, i32)) = tuple_init (%.loc5_60.3, %.loc5_60.5) to file.%b_ref.var [template = constants.%tuple.4]
// CHECK:STDOUT:   %.loc5_61: init (((i32,), i32), (i32, i32)) = converted %.loc5_60.1, %.loc5_60.6 [template = constants.%tuple.4]
// CHECK:STDOUT:   assign file.%b_ref.var, %.loc5_61
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: type = tuple_type ((type,), type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (((type,), type), (type, type)) [template]
// CHECK:STDOUT:   %.6: type = tuple_type ((i32,), i32) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.8: type = tuple_type (((i32,), i32), (i32, i32)) [template]
// CHECK:STDOUT:   %.9: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.10: type = ptr_type ((i32,), i32) [template]
// CHECK:STDOUT:   %.11: type = tuple_type (((i32,), i32)*, (i32, i32)*) [template]
// CHECK:STDOUT:   %.12: type = ptr_type (((i32,), i32)*, (i32, i32)*) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .a_ref = %import_ref.1
// CHECK:STDOUT:     .b_ref = %import_ref.2
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .a = %a
// CHECK:STDOUT:     .b = %b
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: ref (i32,) = import_ref ir0, inst+7, loaded
// CHECK:STDOUT:   %import_ref.2: ref (((i32,), i32), (i32, i32)) = import_ref ir0, inst+35, loaded
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %.loc4_13.1: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc4_13.2: type = converted %.loc4_13.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %a.var: ref (i32,) = var a
// CHECK:STDOUT:   %a: ref (i32,) = bind_name a, %a.var
// CHECK:STDOUT:   %.loc5_15: (type,) = tuple_literal (i32)
// CHECK:STDOUT:   %.loc5_21: ((type,), type) = tuple_literal (%.loc5_15, i32)
// CHECK:STDOUT:   %.loc5_33: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc5_34.1: (((type,), type), (type, type)) = tuple_literal (%.loc5_21, %.loc5_33)
// CHECK:STDOUT:   %.loc5_34.2: type = converted %.loc5_15, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc5_34.3: type = converted %.loc5_21, constants.%.6 [template = constants.%.6]
// CHECK:STDOUT:   %.loc5_34.4: type = converted %.loc5_33, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:   %.loc5_34.5: type = converted %.loc5_34.1, constants.%.8 [template = constants.%.8]
// CHECK:STDOUT:   %b.var: ref (((i32,), i32), (i32, i32)) = var b
// CHECK:STDOUT:   %b: ref (((i32,), i32), (i32, i32)) = bind_name b, %b.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref (i32,) = name_ref a_ref, file.%import_ref.1
// CHECK:STDOUT:   %.loc4_17.1: ref i32 = tuple_access %a_ref.ref, element0
// CHECK:STDOUT:   %.loc4_17.2: i32 = bind_value %.loc4_17.1
// CHECK:STDOUT:   %.loc4_17.3: init (i32,) = tuple_init (%.loc4_17.2) to file.%a.var
// CHECK:STDOUT:   %.loc4_22: init (i32,) = converted %a_ref.ref, %.loc4_17.3
// CHECK:STDOUT:   assign file.%a.var, %.loc4_22
// CHECK:STDOUT:   %b_ref.ref: ref (((i32,), i32), (i32, i32)) = name_ref b_ref, file.%import_ref.2
// CHECK:STDOUT:   %.loc5_38.1: ref ((i32,), i32) = tuple_access %b_ref.ref, element0
// CHECK:STDOUT:   %.loc5_38.2: ref (i32,) = tuple_access %.loc5_38.1, element0
// CHECK:STDOUT:   %.loc5_38.3: ref i32 = tuple_access %.loc5_38.2, element0
// CHECK:STDOUT:   %.loc5_38.4: i32 = bind_value %.loc5_38.3
// CHECK:STDOUT:   %.loc5_38.5: ref ((i32,), i32) = tuple_access file.%b.var, element0
// CHECK:STDOUT:   %.loc5_38.6: ref (i32,) = tuple_access %.loc5_38.5, element0
// CHECK:STDOUT:   %.loc5_38.7: init (i32,) = tuple_init (%.loc5_38.4) to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.8: init (i32,) = converted %.loc5_38.2, %.loc5_38.7
// CHECK:STDOUT:   %.loc5_38.9: init (i32,) = initialize_from %.loc5_38.8 to %.loc5_38.6
// CHECK:STDOUT:   %.loc5_38.10: ref i32 = tuple_access %.loc5_38.1, element1
// CHECK:STDOUT:   %.loc5_38.11: i32 = bind_value %.loc5_38.10
// CHECK:STDOUT:   %.loc5_38.12: ref i32 = tuple_access %.loc5_38.5, element1
// CHECK:STDOUT:   %.loc5_38.13: init i32 = initialize_from %.loc5_38.11 to %.loc5_38.12
// CHECK:STDOUT:   %.loc5_38.14: init ((i32,), i32) = tuple_init (%.loc5_38.9, %.loc5_38.13) to %.loc5_38.5
// CHECK:STDOUT:   %.loc5_38.15: init ((i32,), i32) = converted %.loc5_38.1, %.loc5_38.14
// CHECK:STDOUT:   %.loc5_38.16: ref (i32, i32) = tuple_access %b_ref.ref, element1
// CHECK:STDOUT:   %.loc5_38.17: ref i32 = tuple_access %.loc5_38.16, element0
// CHECK:STDOUT:   %.loc5_38.18: i32 = bind_value %.loc5_38.17
// CHECK:STDOUT:   %.loc5_38.19: ref (i32, i32) = tuple_access file.%b.var, element1
// CHECK:STDOUT:   %.loc5_38.20: ref i32 = tuple_access %.loc5_38.19, element0
// CHECK:STDOUT:   %.loc5_38.21: init i32 = initialize_from %.loc5_38.18 to %.loc5_38.20
// CHECK:STDOUT:   %.loc5_38.22: ref i32 = tuple_access %.loc5_38.16, element1
// CHECK:STDOUT:   %.loc5_38.23: i32 = bind_value %.loc5_38.22
// CHECK:STDOUT:   %.loc5_38.24: ref i32 = tuple_access %.loc5_38.19, element1
// CHECK:STDOUT:   %.loc5_38.25: init i32 = initialize_from %.loc5_38.23 to %.loc5_38.24
// CHECK:STDOUT:   %.loc5_38.26: init (i32, i32) = tuple_init (%.loc5_38.21, %.loc5_38.25) to %.loc5_38.19
// CHECK:STDOUT:   %.loc5_38.27: init (i32, i32) = converted %.loc5_38.16, %.loc5_38.26
// CHECK:STDOUT:   %.loc5_38.28: init (((i32,), i32), (i32, i32)) = tuple_init (%.loc5_38.15, %.loc5_38.27) to file.%b.var
// CHECK:STDOUT:   %.loc5_43: init (((i32,), i32), (i32, i32)) = converted %b_ref.ref, %.loc5_38.28
// CHECK:STDOUT:   assign file.%b.var, %.loc5_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
