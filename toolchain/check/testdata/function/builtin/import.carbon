// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// --- core.carbon

package Core api;

import library "prelude";

fn Add(a: i32, b: i32) -> i32 = "int.sadd";

// --- use.carbon

import Core;

var arr: [i32; Core.Add(1, 2)] = (1, 2, 3);

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Add: type = fn_type @Add [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: Add = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Add = %Add.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Add.decl: Add = fn_decl @Add [template = constants.%struct] {
// CHECK:STDOUT:     %a.loc6_8.1: i32 = param a
// CHECK:STDOUT:     @Add.%a: i32 = bind_name a, %a.loc6_8.1
// CHECK:STDOUT:     %b.loc6_16.1: i32 = param b
// CHECK:STDOUT:     @Add.%b: i32 = bind_name b, %b.loc6_16.1
// CHECK:STDOUT:     @Add.%return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: --- use.carbon
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
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Core.ref: <namespace> = name_ref Core, %Core [template = %Core]
// CHECK:STDOUT:   %import_ref: Add = import_ref ir2, inst+6, loaded [template = constants.%struct]
// CHECK:STDOUT:   %Add.ref: Add = name_ref Add, %import_ref [template = constants.%struct]
// CHECK:STDOUT:   %.loc4_25: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_28: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %int.sadd: init i32 = call %Add.ref(%.loc4_25, %.loc4_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_30: type = array_type %int.sadd, i32 [template = constants.%.5]
// CHECK:STDOUT:   %arr.var: ref [i32; 3] = var arr
// CHECK:STDOUT:   %arr: ref [i32; 3] = bind_name arr, %arr.var
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Add(%a: i32, %b: i32) -> i32 = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc4_35: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_38: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_41: i32 = int_literal 3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_42.1: (i32, i32, i32) = tuple_literal (%.loc4_35, %.loc4_38, %.loc4_41)
// CHECK:STDOUT:   %.loc4_42.2: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc4_42.3: ref i32 = array_index file.%arr.var, %.loc4_42.2
// CHECK:STDOUT:   %.loc4_42.4: init i32 = initialize_from %.loc4_35 to %.loc4_42.3 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_42.5: i32 = int_literal 1 [template = constants.%.2]
// CHECK:STDOUT:   %.loc4_42.6: ref i32 = array_index file.%arr.var, %.loc4_42.5
// CHECK:STDOUT:   %.loc4_42.7: init i32 = initialize_from %.loc4_38 to %.loc4_42.6 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_42.8: i32 = int_literal 2 [template = constants.%.3]
// CHECK:STDOUT:   %.loc4_42.9: ref i32 = array_index file.%arr.var, %.loc4_42.8
// CHECK:STDOUT:   %.loc4_42.10: init i32 = initialize_from %.loc4_41 to %.loc4_42.9 [template = constants.%.4]
// CHECK:STDOUT:   %.loc4_42.11: init [i32; 3] = array_init (%.loc4_42.4, %.loc4_42.7, %.loc4_42.10) to file.%arr.var [template = constants.%array]
// CHECK:STDOUT:   %.loc4_43: init [i32; 3] = converted %.loc4_42.1, %.loc4_42.11 [template = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
