// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Ptr(p: i32*) -> i32* {
  return p;
}

fn ConstPtr(p: const i32*) -> (const i32)* {
  return p;
}

// CHECK:STDOUT: --- types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Ptr = %Ptr
// CHECK:STDOUT:     .ConstPtr = %ConstPtr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Ptr: <function> = fn_decl @Ptr [template] {
// CHECK:STDOUT:     %.loc7_14: type = ptr_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %p.loc7_8.1: i32* = param p
// CHECK:STDOUT:     @Ptr.%p: i32* = bind_name p, %p.loc7_8.1
// CHECK:STDOUT:     %.loc7_23: type = ptr_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %return.var.loc7: ref i32* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ConstPtr: <function> = fn_decl @ConstPtr [template] {
// CHECK:STDOUT:     %.loc11_16: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_25: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:     %p.loc11_13.1: const i32* = param p
// CHECK:STDOUT:     @ConstPtr.%p: const i32* = bind_name p, %p.loc11_13.1
// CHECK:STDOUT:     %.loc11_32: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_42: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:     %return.var.loc11: ref const i32* = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Ptr(%p: i32*) -> i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: i32* = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstPtr(%p: const i32*) -> const i32* {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32* = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
