// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F() -> (i32, i32);

fn G() -> (i32, i32) {
  var v: (i32, i32) = F();
  v = F();
  return F();
}

fn H() -> i32 {
  return G()[0];
}

// CHECK:STDOUT: --- in_place_tuple_init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.2: type = tuple_type (i32, i32) [template]
// CHECK:STDOUT:   %.3: type = ptr_type (i32, i32) [template]
// CHECK:STDOUT:   %.4: i32 = int_literal 0 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F, .G = %G, .H = %H} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template]
// CHECK:STDOUT:   %H: <function> = fn_decl @H [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() -> %return: (i32, i32);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %return: (i32, i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc10_19.1: (type, type) = tuple_literal (i32, i32)
// CHECK:STDOUT:   %.loc10_19.2: type = converted %.loc10_19.1, constants.%.2 [template = constants.%.2]
// CHECK:STDOUT:   %v.var: ref (i32, i32) = var v
// CHECK:STDOUT:   %v: ref (i32, i32) = bind_name v, %v.var
// CHECK:STDOUT:   %F.ref.loc10: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc10_7: ref (i32, i32) = splice_block %v.var {}
// CHECK:STDOUT:   %.loc10_24: init (i32, i32) = call %F.ref.loc10() to %.loc10_7
// CHECK:STDOUT:   assign %v.var, %.loc10_24
// CHECK:STDOUT:   %v.ref: ref (i32, i32) = name_ref v, %v
// CHECK:STDOUT:   %F.ref.loc11: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc11_3: ref (i32, i32) = splice_block %v.ref {}
// CHECK:STDOUT:   %.loc11_8: init (i32, i32) = call %F.ref.loc11() to %.loc11_3
// CHECK:STDOUT:   assign %v.ref, %.loc11_8
// CHECK:STDOUT:   %F.ref.loc12: <function> = name_ref F, file.%F [template = file.%F]
// CHECK:STDOUT:   %.loc9: ref (i32, i32) = splice_block %return {}
// CHECK:STDOUT:   %.loc12: init (i32, i32) = call %F.ref.loc12() to %.loc9
// CHECK:STDOUT:   return %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %G.ref: <function> = name_ref G, file.%G [template = file.%G]
// CHECK:STDOUT:   %.loc16_11.1: ref (i32, i32) = temporary_storage
// CHECK:STDOUT:   %.loc16_11.2: init (i32, i32) = call %G.ref() to %.loc16_11.1
// CHECK:STDOUT:   %.loc16_14: i32 = int_literal 0 [template = constants.%.4]
// CHECK:STDOUT:   %.loc16_11.3: ref (i32, i32) = temporary %.loc16_11.1, %.loc16_11.2
// CHECK:STDOUT:   %.loc16_15.1: ref i32 = tuple_index %.loc16_11.3, %.loc16_14
// CHECK:STDOUT:   %.loc16_15.2: i32 = bind_value %.loc16_15.1
// CHECK:STDOUT:   return %.loc16_15.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
