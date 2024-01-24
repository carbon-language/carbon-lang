// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32,) { return (0,); }

fn Run() {
  var t: [i32; 1] = F();
}

// CHECK:STDOUT: --- assign_return_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.3: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.4: (i32,) = tuple_value (%.3) [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.6: type = array_type %.5, i32 [template]
// CHECK:STDOUT:   %.7: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.8: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .Run = %Run} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %Run: <function> = fn_decl @Run [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> (i32,) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_28: i32 = int_literal 0 [template = constants.%.3]
// CHECK:STDOUT:   %.loc7_30.1: (i32,) = tuple_literal (%.loc7_28)
// CHECK:STDOUT:   %.loc7_30.2: (i32,) = tuple_value (%.loc7_28) [template = constants.%.4]
// CHECK:STDOUT:   %.loc7_30.3: (i32,) = converted %.loc7_30.1, %.loc7_30.2 [template = constants.%.4]
// CHECK:STDOUT:   return %.loc7_30.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Run() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_16: i32 = int_literal 1 [template = constants.%.5]
// CHECK:STDOUT:   %.loc10_17: type = array_type %.loc10_16, i32 [template = constants.%.6]
// CHECK:STDOUT:   %t.var: ref [i32; 1] = var t
// CHECK:STDOUT:   %t: ref [i32; 1] = bind_name t, %t.var
// CHECK:STDOUT:   %F.ref: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc10_22.1: init (i32,) = call %F.ref()
// CHECK:STDOUT:   %.loc10_22.2: ref (i32,) = temporary_storage
// CHECK:STDOUT:   %.loc10_22.3: ref (i32,) = temporary %.loc10_22.2, %.loc10_22.1
// CHECK:STDOUT:   %.loc10_22.4: ref i32 = tuple_access %.loc10_22.3, element0
// CHECK:STDOUT:   %.loc10_22.5: i32 = bind_value %.loc10_22.4
// CHECK:STDOUT:   %.loc10_22.6: i32 = int_literal 0 [template = constants.%.8]
// CHECK:STDOUT:   %.loc10_22.7: ref i32 = array_index %t.var, %.loc10_22.6
// CHECK:STDOUT:   %.loc10_22.8: init i32 = initialize_from %.loc10_22.5 to %.loc10_22.7
// CHECK:STDOUT:   %.loc10_22.9: init [i32; 1] = array_init (%.loc10_22.8) to %t.var
// CHECK:STDOUT:   %.loc10_22.10: init [i32; 1] = converted %.loc10_22.1, %.loc10_22.9
// CHECK:STDOUT:   assign %t.var, %.loc10_22.10
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
