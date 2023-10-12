// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: file "nested_const.carbon" {
// CHECK:STDOUT:   %F: <function> = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: const (const (const i32*)*)) -> const i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const (const (const i32*)*) = name_reference "p", %p
// CHECK:STDOUT:   %.loc9_11.1: ref const (const i32*) = dereference %p.ref
// CHECK:STDOUT:   %.loc9_11.2: const (const i32*) = bind_value %.loc9_11.1
// CHECK:STDOUT:   %.loc9_10.1: ref const i32 = dereference %.loc9_11.2
// CHECK:STDOUT:   %.loc9_10.2: const i32 = bind_value %.loc9_10.1
// CHECK:STDOUT:   return %.loc9_10.2
// CHECK:STDOUT: }
