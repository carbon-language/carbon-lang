// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// Check that we propagate the `symbolic` tag through evaluations.
fn F(T:! type) {
  var u: (T*, const T);
  var v: {.a: T};
  var w: [T; 5];
}

// CHECK:STDOUT: --- symbolic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %.2: type = const_type T [symbolic]
// CHECK:STDOUT:   %.3: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.4: type = tuple_type (T*, const T) [symbolic]
// CHECK:STDOUT:   %.5: type = tuple_type (T*, T) [symbolic]
// CHECK:STDOUT:   %.6: type = ptr_type (T*, T) [symbolic]
// CHECK:STDOUT:   %.7: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT:   %.8: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.9: type = array_type %.8, T [symbolic]
// CHECK:STDOUT:   %.10: type = ptr_type [T; 5] [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref.loc9_11: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc9_12: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %T.ref.loc9_21: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc9_15: type = const_type T [symbolic = constants.%.2]
// CHECK:STDOUT:   %.loc9_22.1: (type, type) = tuple_literal (%.loc9_12, %.loc9_15)
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_22.1, constants.%.4 [symbolic = constants.%.4]
// CHECK:STDOUT:   %u.var: ref (T*, const T) = var u
// CHECK:STDOUT:   %u: ref (T*, const T) = bind_name u, %u.var
// CHECK:STDOUT:   %T.ref.loc10: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc10: type = struct_type {.a: T} [symbolic = constants.%.7]
// CHECK:STDOUT:   %v.var: ref {.a: T} = var v
// CHECK:STDOUT:   %v: ref {.a: T} = bind_name v, %v.var
// CHECK:STDOUT:   %T.ref.loc11: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, T [symbolic = constants.%.9]
// CHECK:STDOUT:   %w.var: ref [T; 5] = var w
// CHECK:STDOUT:   %w: ref [T; 5] = bind_name w, %w.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
