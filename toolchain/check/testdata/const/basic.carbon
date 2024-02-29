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
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: type = const_type i32* [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .A = %A
// CHECK:STDOUT:     .B = %B
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %A: <function> = fn_decl @A [template] {
// CHECK:STDOUT:     %.loc7_9: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_18: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc7_19: type = ptr_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     %p.loc7_6.1: const i32** = param p
// CHECK:STDOUT:     @A.%p: const i32** = bind_name p, %p.loc7_6.1
// CHECK:STDOUT:     %.loc7_25: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc7_34: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc7_35: type = ptr_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc7: ref const i32** = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B: <function> = fn_decl @B [template] {
// CHECK:STDOUT:     %.loc11_19: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_9: type = const_type i32* [template = constants.%.5]
// CHECK:STDOUT:     %p.loc11_6.1: const (i32*) = param p
// CHECK:STDOUT:     @B.%p: const (i32*) = bind_name p, %p.loc11_6.1
// CHECK:STDOUT:     %.loc11_36: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_26: type = const_type i32* [template = constants.%.5]
// CHECK:STDOUT:     %return.var.loc11: ref const (i32*) = var <return slot>
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
