// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn ConstMismatch(p: const {}*) -> const ({}*) {
  // CHECK:STDERR: fail_type_mismatch.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `const {}*` to `const ({}*)`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_type_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: type = const_type {} [template]
// CHECK:STDOUT:   %.3: type = ptr_type const {} [template]
// CHECK:STDOUT:   %.4: type = ptr_type {} [template]
// CHECK:STDOUT:   %.5: type = const_type {}* [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.ConstMismatch = %ConstMismatch} [template]
// CHECK:STDOUT:   %ConstMismatch: <function> = fn_decl @ConstMismatch [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p: const {}*) -> const ({}*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const {}* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
