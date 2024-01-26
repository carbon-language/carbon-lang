// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(T:! type, U:! type) {
  var p: T*;
  // CHECK:STDERR: fail_type_param_mismatch.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `T` to `U`.
  // CHECK:STDERR:   let n: U = *p;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  let n: U = *p;
}

// CHECK:STDOUT: --- fail_type_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.F = %F} [template]
// CHECK:STDOUT:   %F: <function> = fn_decl @F [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %U: type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %T.ref: type = name_ref T, %T [symbolic = %T]
// CHECK:STDOUT:   %.loc8: type = ptr_type T [symbolic = constants.%.1]
// CHECK:STDOUT:   %p.var: ref T* = var p
// CHECK:STDOUT:   %p: ref T* = bind_name p, %p.var
// CHECK:STDOUT:   %U.ref: type = name_ref U, %U [symbolic = %U]
// CHECK:STDOUT:   %p.ref: ref T* = name_ref p, %p
// CHECK:STDOUT:   %.loc12_15: T* = bind_value %p.ref
// CHECK:STDOUT:   %.loc12_14: ref T = deref %.loc12_15
// CHECK:STDOUT:   %n: U = bind_name n, <error>
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
