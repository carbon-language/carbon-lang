// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

// CHECK:STDERR: fail_collapse.carbon:[[@LINE+4]]:9: WARNING: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn G(p: const (const i32)**) -> i32** {
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn G(p: const (const i32)**) -> i32** {
  // CHECK:STDERR: fail_collapse.carbon:[[@LINE+3]]:3: ERROR: Cannot implicitly convert from `const i32**` to `i32**`.
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  return p;
}

// CHECK:STDOUT: --- fail_collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Int32 = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type const i32 [template]
// CHECK:STDOUT:   %.4: type = ptr_type const i32* [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type i32* [template]
// CHECK:STDOUT:   %G: type = fn_type @G [template]
// CHECK:STDOUT:   %struct.2: G = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %import_ref.1: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %import_ref.2: Int32 = import_ref ir3, inst+3, loaded [template = constants.%struct.1]
// CHECK:STDOUT:   %G.decl: G = fn_decl @G [template = constants.%struct.2] {
// CHECK:STDOUT:     %int.make_type_32.loc11_22: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_16.1: type = value_of_initializer %int.make_type_32.loc11_22 [template = i32]
// CHECK:STDOUT:     %.loc11_16.2: type = converted %int.make_type_32.loc11_22, %.loc11_16.1 [template = i32]
// CHECK:STDOUT:     %.loc11_16.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_9: type = const_type const i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_26: type = ptr_type const i32 [template = constants.%.3]
// CHECK:STDOUT:     %.loc11_27: type = ptr_type const i32* [template = constants.%.4]
// CHECK:STDOUT:     %p.loc11_6.1: const i32** = param p
// CHECK:STDOUT:     @G.%p: const i32** = bind_name p, %p.loc11_6.1
// CHECK:STDOUT:     %int.make_type_32.loc11_33: init type = call constants.%struct.1() [template = i32]
// CHECK:STDOUT:     %.loc11_36.1: type = value_of_initializer %int.make_type_32.loc11_33 [template = i32]
// CHECK:STDOUT:     %.loc11_36.2: type = converted %int.make_type_32.loc11_33, %.loc11_36.1 [template = i32]
// CHECK:STDOUT:     %.loc11_36.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc11_37: type = ptr_type i32* [template = constants.%.6]
// CHECK:STDOUT:     @G.%return: ref i32** = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: const i32**) -> i32** {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const i32** = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
