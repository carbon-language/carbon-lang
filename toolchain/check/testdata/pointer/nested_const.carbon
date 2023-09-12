// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// CHECK:STDOUT: file "nested_const.carbon" {

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
// CHECK:STDOUT:   %.loc[[@LINE-1]] = fn_decl @F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: const (const (const i32*)*)) -> const i32 {
// CHECK:STDOUT: !entry:
  return **p;
  // CHECK:STDOUT:   %.loc[[@LINE-1]]_11.1: ref const (const i32*) = dereference %p
  // CHECK:STDOUT:   %.loc[[@LINE-2]]_11.2: const (const i32*) = bind_value %.loc[[@LINE-2]]_11.1
  // CHECK:STDOUT:   %.loc[[@LINE-3]]_10.1: ref const i32 = dereference %.loc[[@LINE-3]]_11.2
  // CHECK:STDOUT:   %.loc[[@LINE-4]]_10.2: const i32 = bind_value %.loc[[@LINE-4]]_10.1
  // CHECK:STDOUT:   return %.loc[[@LINE-5]]_10.2
  // CHECK:STDOUT: }
}
