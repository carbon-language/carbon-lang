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
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 2 [template]
// CHECK:STDOUT:   %.5: type = array_type %.4, (i32, i32, i32) [template]
// CHECK:STDOUT:   %.6: type = ptr_type [(i32, i32, i32); 2] [template]
// CHECK:STDOUT:   %.7: type = tuple_type ((i32, i32, i32), (i32, i32, i32)) [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.9: i32 = int_literal 1 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .G = %G} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_25.1: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 2 [template = constants.%.4]
// CHECK:STDOUT:   %.loc10_25.2: type = converted %.loc10_25.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %.loc10_29: type = array_type %.loc10_28, (i32, i32, i32) [template = constants.%.5]
// CHECK:STDOUT:   %v.var: ref [(i32, i32, i32); 2] = var v
// CHECK:STDOUT:   %v: ref [(i32, i32, i32); 2] = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10_34: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc10_42.3: ref (i32, i32, i32) = splice_block %.loc10_42.2 {
// CHECK:STDOUT:     %.loc10_42.1: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:     %.loc10_42.2: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_35: init (i32, i32, i32) = call %F.ref.loc10_34() to %.loc10_42.3
// CHECK:STDOUT:   %F.ref.loc10_39: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc10_42.6: ref (i32, i32, i32) = splice_block %.loc10_42.5 {
// CHECK:STDOUT:     %.loc10_42.4: i32 = int_literal 1 [template = constants.%.9]
// CHECK:STDOUT:     %.loc10_42.5: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_40: init (i32, i32, i32) = call %F.ref.loc10_39() to %.loc10_42.6
// CHECK:STDOUT:   %.loc10_42.7: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_35, %.loc10_40)
// CHECK:STDOUT:   %.loc10_42.8: init [(i32, i32, i32); 2] = array_init (%.loc10_35, %.loc10_40) to %v.var
// CHECK:STDOUT:   %.loc10_42.9: init [(i32, i32, i32); 2] = converted %.loc10_42.7, %.loc10_42.8
// CHECK:STDOUT:   assign %v.var, %.loc10_42.9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
