// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn A(p: const i32**) -> const i32** {
  return p;
}

fn B(p: const (i32*)) -> const (i32*) {
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32* [template]
// CHECK:STDOUT:   %A: type = fn_type @A [template]
// CHECK:STDOUT:   %.4: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: A = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: type = const_type i32* [template]
// CHECK:STDOUT:   %B: type = fn_type @B [template]
// CHECK:STDOUT:   %struct.2: B = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %A.decl: A = fn_decl @A [template = constants.%struct.1] {
// CHECK:STDOUT:     %.loc7_9: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_18: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc7_19: type = ptr_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     %p.loc7_6.1: const i32** = param p
// CHECK:STDOUT:     @A.%p: const i32** = bind_name p, %p.loc7_6.1
// CHECK:STDOUT:     %.loc7_25: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_34: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc7_35: type = ptr_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     @A.%return: ref const i32** = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.decl: B = fn_decl @B [template = constants.%struct.2] {
// CHECK:STDOUT:     %.loc11_19: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_9: type = const_type i32* [template = constants.%.6]
// CHECK:STDOUT:     %p.loc11_6.1: const (i32*) = param p
// CHECK:STDOUT:     @B.%p: const (i32*) = bind_name p, %p.loc11_6.1
// CHECK:STDOUT:     %.loc11_36: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_26: type = const_type i32* [template = constants.%.6]
// CHECK:STDOUT:     @B.%return: ref const (i32*) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p: const i32**) -> const i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32** = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p: const (i32*)) -> const (i32*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const (i32*) = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
