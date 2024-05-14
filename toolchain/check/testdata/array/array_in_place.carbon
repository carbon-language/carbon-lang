// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: --- array_in_place.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: F = struct_value () [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, (i32, i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.7: type = ptr_type [(i32, i32, i32); 2] [template]
// CHECK:STDOUT:   %.8: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.10: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_25.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:     %.loc7_25.2: type = converted %.loc7_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref (i32, i32, i32) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_25.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25.2: type = converted %.loc10_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_29: type = array_type %.loc10_28, (i32, i32, i32) [template = constants.%.5]
// CHECK:STDOUT:   %v.var: ref [(i32, i32, i32); 2] = var v
// CHECK:STDOUT:   %v: ref [(i32, i32, i32); 2] = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10_34: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_42.3: ref (i32, i32, i32) = splice_block %.loc10_42.2 {
// CHECK:STDOUT:     %.loc10_42.1: i32 = int_literal 0 [template = constants.%.9]
// CHECK:STDOUT:     %.loc10_42.2: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_35: init (i32, i32, i32) = call %F.ref.loc10_34() to %.loc10_42.3
// CHECK:STDOUT:   %F.ref.loc10_39: F = name_ref F, file.%F.decl [template = constants.%struct.1]
// CHECK:STDOUT:   %.loc10_42.6: ref (i32, i32, i32) = splice_block %.loc10_42.5 {
// CHECK:STDOUT:     %.loc10_42.4: i32 = int_literal 1 [template = constants.%.10]
// CHECK:STDOUT:     %.loc10_42.5: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.call.loc10_40: init (i32, i32, i32) = call %F.ref.loc10_39() to %.loc10_42.6
// CHECK:STDOUT:   %.loc10_42.7: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%F.call.loc10_35, %F.call.loc10_40)
// CHECK:STDOUT:   %.loc10_42.8: init [(i32, i32, i32); 2] = array_init (%F.call.loc10_35, %F.call.loc10_40) to %v.var
// CHECK:STDOUT:   %.loc10_43: init [(i32, i32, i32); 2] = converted %.loc10_42.7, %.loc10_42.8
// CHECK:STDOUT:   assign %v.var, %.loc10_43
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
