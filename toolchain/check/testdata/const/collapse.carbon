// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/collapse.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/collapse.carbon

// OK, `const (const i32)` is the same type as `const i32`.
// CHECK:STDERR: collapse.carbon:[[@LINE+3]]:25: WARNING: `const` applied repeatedly to the same type has no additional effect.
// CHECK:STDERR: fn F(p: const i32**) -> const (const i32)** {
// CHECK:STDERR:                         ^~~~~~~~~~~~~~~~~
fn F(p: const i32**) -> const (const i32)** {
  return p;
}

// CHECK:STDOUT: --- collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
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
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %int.make_type_32.loc15_15: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_9.1: type = value_of_initializer %int.make_type_32.loc15_15 [template = i32]
// CHECK:STDOUT:     %.loc15_9.2: type = converted %int.make_type_32.loc15_15, %.loc15_9.1 [template = i32]
// CHECK:STDOUT:     %.loc15_9.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_18: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_19: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %p.loc15_6.1: %.4 = param p
// CHECK:STDOUT:     @F.%p: %.4 = bind_name p, %p.loc15_6.1
// CHECK:STDOUT:     %int.make_type_32.loc15_38: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc15_32.1: type = value_of_initializer %int.make_type_32.loc15_38 [template = i32]
// CHECK:STDOUT:     %.loc15_32.2: type = converted %int.make_type_32.loc15_38, %.loc15_32.1 [template = i32]
// CHECK:STDOUT:     %.loc15_32.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_25: type = const_type %.2 [template = constants.%.2]
// CHECK:STDOUT:     %.loc15_42: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc15_43: type = ptr_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     @F.%return: ref %.4 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: %.4) -> %.4 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.4 = name_ref p, %p
// CHECK:STDOUT:   return %p.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
