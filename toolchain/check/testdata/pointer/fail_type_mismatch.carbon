// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/fail_type_mismatch.carbon

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
// CHECK:STDOUT:   %ConstMismatch: type = fn_type @ConstMismatch [template]
// CHECK:STDOUT:   %.6: type = tuple_type () [template]
// CHECK:STDOUT:   %struct: ConstMismatch = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .ConstMismatch = %ConstMismatch.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %ConstMismatch.decl: ConstMismatch = fn_decl @ConstMismatch [template = constants.%struct] {
// CHECK:STDOUT:     %.loc11_28: {} = struct_literal ()
// CHECK:STDOUT:     %.loc11_21.1: type = converted %.loc11_28, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_21.2: type = const_type {} [template = constants.%.2]
// CHECK:STDOUT:     %.loc11_29: type = ptr_type const {} [template = constants.%.3]
// CHECK:STDOUT:     %p.loc11_18.1: const {}* = param p
// CHECK:STDOUT:     @ConstMismatch.%p: const {}* = bind_name p, %p.loc11_18.1
// CHECK:STDOUT:     %.loc11_43: {} = struct_literal ()
// CHECK:STDOUT:     %.loc11_44.1: type = converted %.loc11_43, constants.%.1 [template = constants.%.1]
// CHECK:STDOUT:     %.loc11_44.2: type = ptr_type {} [template = constants.%.4]
// CHECK:STDOUT:     %.loc11_35: type = const_type {}* [template = constants.%.5]
// CHECK:STDOUT:     @ConstMismatch.%return: ref const ({}*) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ConstMismatch(%p: const {}*) -> const ({}*) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: const {}* = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
