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
// CHECK:STDOUT:   package: <namespace> = namespace {.Ptr = %Ptr, .ConstPtr = %ConstPtr} [template]
// CHECK:STDOUT:   %Ptr: <function> = fn_decl @Ptr [template]
// CHECK:STDOUT:   %ConstPtr: <function> = fn_decl @ConstPtr [template]
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
