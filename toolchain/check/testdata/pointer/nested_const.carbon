// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: --- nested_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %.3: type = const_type const i32* [template]
// CHECK:STDOUT:   %.4: type = ptr_type const (const i32*) [template]
// CHECK:STDOUT:   %.5: type = const_type const (const i32*)* [template]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: F = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct] {
// CHECK:STDOUT:     %.loc8_23: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc8_32: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc8_16: type = const_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     %.loc8_34: type = ptr_type const (const i32*) [template = constants.%.4]
// CHECK:STDOUT:     %.loc8_9: type = const_type const (const i32*)* [template = constants.%.5]
// CHECK:STDOUT:     %p.loc8_6.1: const (const (const i32*)*) = param p
// CHECK:STDOUT:     @F.%p: const (const (const i32*)*) = bind_name p, %p.loc8_6.1
// CHECK:STDOUT:     %.loc8_41: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     @F.%return: ref const i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: const (const (const i32*)*)) -> const i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const (const (const i32*)*) = name_ref p, %p
// CHECK:STDOUT:   %.loc9_11.1: ref const (const i32*) = deref %p.ref
// CHECK:STDOUT:   %.loc9_11.2: const (const i32*) = bind_value %.loc9_11.1
// CHECK:STDOUT:   %.loc9_10.1: ref const i32 = deref %.loc9_11.2
// CHECK:STDOUT:   %.loc9_10.2: const i32 = bind_value %.loc9_10.1
// CHECK:STDOUT:   return %.loc9_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
