// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

var a: [i32; Add(1, 2)] = (1, 2, 3);
let b: [i32; 3]* = &a;

// CHECK:STDOUT: --- canonicalize_index.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Add: type = fn_type @Add [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Add = struct_value () [template]
// CHECK:STDOUT:   %.2: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 3 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type [i32; 3] [template]
// CHECK:STDOUT:   %.7: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %array: [i32; 3] = tuple_value (%.2, %.3, %.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:     .a = %a.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Add.decl: Add = fn_decl @Add [template = constants.%struct] {
// CHECK:STDOUT:     %a.loc7_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc7_8.1
// CHECK:STDOUT:     %b.loc7_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc7_16.1
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.ref: Add = name_ref Add, %Add.decl [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_18: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_21: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc9_18, %.loc9_21) [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_23: type = array_type %int.sadd, i32 [template = constants.%.5]
// CHECK:STDOUT:   %a.var: ref [i32; 3] = var a
// CHECK:STDOUT:   %a.loc9: ref [i32; 3] = bind_name a, %a.var
// CHECK:STDOUT:   %.loc10_14: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_15: type = array_type %.loc10_14, i32 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_16: type = ptr_type [i32; 3] [template = constants.%.6]
// CHECK:STDOUT:   %a.ref: ref [i32; 3] = name_ref a, %a.loc9
// CHECK:STDOUT:   %.loc10_20: [i32; 3]* = addr_of %a.ref
// CHECK:STDOUT:   %b.loc10: [i32; 3]* = bind_name b, %.loc10_20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc9_28: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_31: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_34: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_35.1: (i32, i32, i32) = tuple_literal (%.loc9_28, %.loc9_31, %.loc9_34)
// CHECK:STDOUT:   %.loc9_35.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc9_35.3: ref i32 = array_index file.%a.var, %.loc9_35.2
// CHECK:STDOUT:   %.loc9_35.4: init i32 = initialize_from %.loc9_28 to %.loc9_35.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_35.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc9_35.6: ref i32 = array_index file.%a.var, %.loc9_35.5
// CHECK:STDOUT:   %.loc9_35.7: init i32 = initialize_from %.loc9_31 to %.loc9_35.6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_35.8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc9_35.9: ref i32 = array_index file.%a.var, %.loc9_35.8
// CHECK:STDOUT:   %.loc9_35.10: init i32 = initialize_from %.loc9_34 to %.loc9_35.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc9_35.11: init [i32; 3] = array_init (%.loc9_35.4, %.loc9_35.7, %.loc9_35.10) to file.%a.var [template = constants.%array]
// CHECK:STDOUT:   %.loc9_36: init [i32; 3] = converted %.loc9_35.1, %.loc9_35.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%a.var, %.loc9_36
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
