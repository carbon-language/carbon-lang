// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(b: bool, n: i32, m: i32) -> i32 {
  var x: [i32; 1] = (0,);
  return if b then x[m] else x[n];
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: i32 = int_literal 1 [template]
// CHECK:STDOUT:   %.2: type = array_type %.1, i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type [i32; 1] [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (i32) [template]
// CHECK:STDOUT:   %.6: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %.7: [i32; 1] = tuple_value (%.4) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b: bool, %n: i32, %m: i32) -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc8_16: i32 = int_literal 1 [template = constants.%.1]
// CHECK:STDOUT:   %.loc8_17: type = array_type %.loc8_16, i32 [template = constants.%.2]
// CHECK:STDOUT:   %x.var: ref [i32; 1] = var x
// CHECK:STDOUT:   %x: ref [i32; 1] = bind_name x, %x.var
// CHECK:STDOUT:   %.loc8_22: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_24.1: (i32,) = tuple_literal (%.loc8_22)
// CHECK:STDOUT:   %.loc8_24.2: i32 = int_literal 0 [template = constants.%.6]
// CHECK:STDOUT:   %.loc8_24.3: ref i32 = array_index %x.var, %.loc8_24.2
// CHECK:STDOUT:   %.loc8_24.4: init i32 = initialize_from %.loc8_22 to %.loc8_24.3 [template = constants.%.4]
// CHECK:STDOUT:   %.loc8_24.5: init [i32; 1] = array_init (%.loc8_24.4) to %x.var [template = constants.%.7]
// CHECK:STDOUT:   %.loc8_24.6: init [i32; 1] = converted %.loc8_24.1, %.loc8_24.5 [template = constants.%.7]
// CHECK:STDOUT:   assign %x.var, %.loc8_24.6
// CHECK:STDOUT:   %b.ref: bool = name_ref b, %b
// CHECK:STDOUT:   if %b.ref br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.then:
// CHECK:STDOUT:   %x.ref.loc9_20: ref [i32; 1] = name_ref x, %x
// CHECK:STDOUT:   %m.ref: i32 = name_ref m, %m
// CHECK:STDOUT:   %.loc9_23.1: ref i32 = array_index %x.ref.loc9_20, %m.ref
// CHECK:STDOUT:   %.loc9_23.2: i32 = bind_value %.loc9_23.1
// CHECK:STDOUT:   br !if.expr.result(%.loc9_23.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.else:
// CHECK:STDOUT:   %x.ref.loc9_30: ref [i32; 1] = name_ref x, %x
// CHECK:STDOUT:   %n.ref: i32 = name_ref n, %n
// CHECK:STDOUT:   %.loc9_33.1: ref i32 = array_index %x.ref.loc9_30, %n.ref
// CHECK:STDOUT:   %.loc9_33.2: i32 = bind_value %.loc9_33.1
// CHECK:STDOUT:   br !if.expr.result(%.loc9_33.2)
// CHECK:STDOUT:
// CHECK:STDOUT: !if.expr.result:
// CHECK:STDOUT:   %.loc9_10: i32 = block_arg !if.expr.result
// CHECK:STDOUT:   return %.loc9_10
// CHECK:STDOUT: }
// CHECK:STDOUT:
