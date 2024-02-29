// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:9: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `const i32**` to `i32**`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = const_type i32 [template]
// CHECK:STDOUT:   %.2: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32* [template]
// CHECK:STDOUT:   %.4: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32* [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .G = %G
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G: <function> = fn_decl @G [template] {
// CHECK:STDOUT:     %.loc10_16: type = const_type i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc10_9: type = const_type const i32 [template = constants.%.1]
// CHECK:STDOUT:     %.loc10_26: type = ptr_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc10_27: type = ptr_type const i32* [template = constants.%.3]
// CHECK:STDOUT:     %p.loc10_6.1: const i32** = param p
// CHECK:STDOUT:     @G.%p: const i32** = bind_name p, %p.loc10_6.1
// CHECK:STDOUT:     %.loc10_36: type = ptr_type i32 [template = constants.%.4]
// CHECK:STDOUT:     %.loc10_37: type = ptr_type i32* [template = constants.%.5]
// CHECK:STDOUT:     %return.var: ref i32** = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: const i32**) -> i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32** = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
