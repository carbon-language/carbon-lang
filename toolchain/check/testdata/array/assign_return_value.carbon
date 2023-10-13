// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: file "assign_return_value.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_30.1: (i32,) = tuple_literal (%.loc7_28)
// CHECK:STDOUT:   %.loc7_30.2: (i32,) = tuple_value %.loc7_30.1, (%.loc7_28)
// CHECK:STDOUT:   return %.loc7_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_17.1: type = array_type %.loc10_16, i32
// CHECK:STDOUT:   %.loc10_17.2: type = ptr_type [i32; 1]
// CHECK:STDOUT:   %t: ref [i32; 1] = var "t"
// CHECK:STDOUT:   %F.ref: <function> = name_reference "F", package.%F
// CHECK:STDOUT:   %.loc10_22.1: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_22.2: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_22.3: ref (i32,) = temporary %.loc10_22.2, %.loc10_22.1
// CHECK:STDOUT:   %.loc10_22.4: ref i32 = tuple_access %.loc10_22.3, member0
// CHECK:STDOUT:   %.loc10_22.5: i32 = bind_value %.loc10_22.4
// CHECK:STDOUT:   %.loc10_22.6: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_22.7: ref i32 = array_index %t, %.loc10_22.6
// CHECK:STDOUT:   %.loc10_22.8: init i32 = initialize_from %.loc10_22.5 to %.loc10_22.7
// CHECK:STDOUT:   %.loc10_22.9: init [i32; 1] = array_init %.loc10_22.3, (%.loc10_22.8) to %t
// CHECK:STDOUT:   assign %t, %.loc10_22.9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
