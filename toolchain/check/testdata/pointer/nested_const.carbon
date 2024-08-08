// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/pointer/nested_const.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/pointer/nested_const.carbon

// TODO: The `const` in the return type should not be necessary.
fn F(p: const (const (const i32*)*)) -> const i32 {
  return **p;
}

// CHECK:STDOUT: --- nested_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = const_type i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.4: type = const_type %.3 [template]
// CHECK:STDOUT:   %.5: type = ptr_type %.4 [template]
// CHECK:STDOUT:   %.6: type = const_type %.5 [template]
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
// CHECK:STDOUT:     %int.make_type_32.loc12_29: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_23.1: type = value_of_initializer %int.make_type_32.loc12_29 [template = i32]
// CHECK:STDOUT:     %.loc12_23.2: type = converted %int.make_type_32.loc12_29, %.loc12_23.1 [template = i32]
// CHECK:STDOUT:     %.loc12_23.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     %.loc12_32: type = ptr_type %.2 [template = constants.%.3]
// CHECK:STDOUT:     %.loc12_16: type = const_type %.3 [template = constants.%.4]
// CHECK:STDOUT:     %.loc12_34: type = ptr_type %.4 [template = constants.%.5]
// CHECK:STDOUT:     %.loc12_9: type = const_type %.5 [template = constants.%.6]
// CHECK:STDOUT:     %p.loc12_6.1: %.6 = param p
// CHECK:STDOUT:     @F.%p: %.6 = bind_name p, %p.loc12_6.1
// CHECK:STDOUT:     %int.make_type_32.loc12_47: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc12_41.1: type = value_of_initializer %int.make_type_32.loc12_47 [template = i32]
// CHECK:STDOUT:     %.loc12_41.2: type = converted %int.make_type_32.loc12_47, %.loc12_41.1 [template = i32]
// CHECK:STDOUT:     %.loc12_41.3: type = const_type i32 [template = constants.%.2]
// CHECK:STDOUT:     @F.%return: ref %.2 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p: %.6) -> %.2 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %.6 = name_ref p, %p
// CHECK:STDOUT:   %.loc13_11.1: ref %.4 = deref %p.ref
// CHECK:STDOUT:   %.loc13_11.2: %.4 = bind_value %.loc13_11.1
// CHECK:STDOUT:   %.loc13_10.1: ref %.2 = deref %.loc13_11.2
// CHECK:STDOUT:   %.loc13_10.2: %.2 = bind_value %.loc13_10.1
// CHECK:STDOUT:   return %.loc13_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
