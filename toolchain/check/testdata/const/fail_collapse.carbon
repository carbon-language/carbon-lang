// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/fail_collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/fail_collapse.carbon

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
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type i32 [template]
// CHECK:STDOUT:   %.6: type = ptr_type %.5 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %.4 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc15_22: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_16.1: type = value_of_initializer %int.make_type_32.loc15_22 [template = i32]
// CHECK:STDOUT:     %.loc15_16.2: type = converted %int.make_type_32.loc15_22, %.loc15_16.1 [template = i32]
// CHECK:STDOUT:     %.loc15_16.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_9: type = const_type %.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_26: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_27: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %p.loc15_6.1: %.4 = param p
// CHECK:STDOUT:     @G.%p: %.4 = bind_name p, %p.loc15_6.1
// CHECK:STDOUT:     %int.make_type_32.loc15_33: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_36.1: type = value_of_initializer %int.make_type_32.loc15_33 [template = i32]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %int.make_type_32.loc15_33, %.loc15_36.1 [template = i32]
// CHECK:STDOUT:     %.loc15_36.3: type = ptr_type i32 [template = constants.%.5]
// CHECK:STDOUT:     %.loc15_37: type = ptr_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     @G.%return: ref %.6 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: %.4) -> %.6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
