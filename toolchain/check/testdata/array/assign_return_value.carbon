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
// CHECK:STDOUT:   %.loc7 = fn_decl @F
// CHECK:STDOUT:   %.loc9 = fn_decl @Run
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28.1: i32 = int_literal 0
// CHECK:STDOUT:   %.loc7_28.2: i32 = stub_reference %.loc7_28.1
// CHECK:STDOUT:   %.loc7_30: (i32,) = tuple_literal (%.loc7_28.2)
// CHECK:STDOUT:   return %.loc7_30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32
// CHECK:STDOUT:   %t: ref [i32; 1] = var "t"
// CHECK:STDOUT:   %.loc10_22.1: init (i32,) = call @F()
// CHECK:STDOUT:   %.loc10_22.2: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_22.3: ref (i32,) = temporary %.loc10_22.2, %.loc10_22.1
// CHECK:STDOUT:   %.loc10_22.4: ref [i32; 1] = stub_reference %t
// CHECK:STDOUT:   %.loc10_22.5: i32 = int_literal 0
// CHECK:STDOUT:   %.loc10_22.6: ref i32 = array_index %.loc10_22.4, %.loc10_22.5
// CHECK:STDOUT:   %.loc10_22.7: ref i32 = tuple_index %.loc10_22.3, %.loc10_22.5
// CHECK:STDOUT:   %.loc10_22.8: i32 = bind_value %.loc10_22.7
// CHECK:STDOUT:   %.loc10_22.9: init i32 = initialize_from %.loc10_22.8 to %.loc10_22.6
// CHECK:STDOUT:   %.loc10_22.10: init [i32; 1] = array_init %.loc10_22.3, (%.loc10_22.9) to %.loc10_22.4
// CHECK:STDOUT:   assign %t, %.loc10_22.10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
