// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32, i32);

fn G() {
  var v: [(i32, i32, i32); 2] = (F(), F());
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc7_25.1: type = tuple_type (type, type, type)
// CHECK:STDOUT:   %.loc7_25.2: type = tuple_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc7_25.3: type = ptr_type (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_29: type = ptr_type [(i32, i32, i32); 2]
// CHECK:STDOUT:   %.loc10_42: type = tuple_type ((i32, i32, i32), (i32, i32, i32))
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "array_in_place.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %G: <function> = fn_decl @G
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_25: (type, type, type) = tuple_literal (i32, i32, i32)
// CHECK:STDOUT:   %.loc10_28: i32 = int_literal 2
// CHECK:STDOUT:   %.loc7: type = converted %.loc10_25, constants.%.loc7_25.2
// CHECK:STDOUT:   %.loc10_29: type = array_type %.loc10_28, (i32, i32, i32)
// CHECK:STDOUT:   %v.var: ref [(i32, i32, i32); 2] = var v
// CHECK:STDOUT:   %v: ref [(i32, i32, i32); 2] = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10_34: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc10_42.3: ref (i32, i32, i32) = splice_block %.loc10_42.2 {
// CHECK:STDOUT:     %.loc10_42.1: i32 = int_literal 0
// CHECK:STDOUT:     %.loc10_42.2: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_35: init (i32, i32, i32) = call %F.ref.loc10_34() to %.loc10_42.3
// CHECK:STDOUT:   %F.ref.loc10_39: <function> = name_ref F, file.%F
// CHECK:STDOUT:   %.loc10_42.6: ref (i32, i32, i32) = splice_block %.loc10_42.5 {
// CHECK:STDOUT:     %.loc10_42.4: i32 = int_literal 1
// CHECK:STDOUT:     %.loc10_42.5: ref (i32, i32, i32) = array_index %v.var, %.loc10_42.4
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10_40: init (i32, i32, i32) = call %F.ref.loc10_39() to %.loc10_42.6
// CHECK:STDOUT:   %.loc10_42.7: ((i32, i32, i32), (i32, i32, i32)) = tuple_literal (%.loc10_35, %.loc10_40)
// CHECK:STDOUT:   %.loc10_42.8: init [(i32, i32, i32); 2] = array_init (%.loc10_35, %.loc10_40) to %v.var
// CHECK:STDOUT:   %.loc10_42.9: init [(i32, i32, i32); 2] = converted %.loc10_42.7, %.loc10_42.8
// CHECK:STDOUT:   assign %v.var, %.loc10_42.9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
