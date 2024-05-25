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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT:   %.2: type = ptr_type T [symbolic]
// CHECK:STDOUT:   %.3: type = const_type T [symbolic]
// CHECK:STDOUT:   %.4: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.5: type = tuple_type (T*, const T) [symbolic]
// CHECK:STDOUT:   %.6: type = tuple_type (T*, T) [symbolic]
// CHECK:STDOUT:   %.7: type = ptr_type (T*, T) [symbolic]
// CHECK:STDOUT:   %.8: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT:   %.9: i32 = int_literal 5 [template]
// CHECK:STDOUT:   %.10: type = array_type %.9, T [symbolic]
// CHECK:STDOUT:   %.11: type = ptr_type [T; 5] [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %T.loc8_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc8_6.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref.loc9_11: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc9_12: type = ptr_type T [symbolic = constants.%.2]
// CHECK:STDOUT:   %T.ref.loc9_21: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc9_15: type = const_type T [symbolic = constants.%.3]
// CHECK:STDOUT:   %.loc9_22.1: (type, type) = tuple_literal (%.loc9_12, %.loc9_15)
// CHECK:STDOUT:   %.loc9_22.2: type = converted %.loc9_22.1, constants.%.5 [symbolic = constants.%.5]
// CHECK:STDOUT:   %u.var: ref (T*, const T) = var u
// CHECK:STDOUT:   %u: ref (T*, const T) = bind_name u, %u.var
// CHECK:STDOUT:   %T.ref.loc10: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc10: type = struct_type {.a: T} [symbolic = constants.%.8]
// CHECK:STDOUT:   %v.var: ref {.a: T} = var v
// CHECK:STDOUT:   %v: ref {.a: T} = bind_name v, %v.var
// CHECK:STDOUT:   %T.ref.loc11: type = name_ref T, %T [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc11_14: i32 = int_literal 5 [template = constants.%.9]
// CHECK:STDOUT:   %.loc11_15: type = array_type %.loc11_14, T [symbolic = constants.%.10]
// CHECK:STDOUT:   %w.var: ref [T; 5] = var w
// CHECK:STDOUT:   %w: ref [T; 5] = bind_name w, %w.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
