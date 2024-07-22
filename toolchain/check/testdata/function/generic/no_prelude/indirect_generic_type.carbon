// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/no_prelude/indirect_generic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/no_prelude/indirect_generic_type.carbon

fn F(T:! type, p: T**) -> T* {
  return *p;
}

// CHECK:STDOUT: --- indirect_generic_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %.1: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %.2: type = ptr_type %.1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.3: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc11_6.1: type = param T
// CHECK:STDOUT:     @F.%T: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = <unexpected>.inst+15 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc11_19: type = name_ref T, @F.%T [symbolic = <unexpected>.inst+15 (constants.%T)]
// CHECK:STDOUT:     %.loc11_20: type = ptr_type %T [symbolic = <unexpected>.inst+16 (constants.%.1)]
// CHECK:STDOUT:     %.loc11_21: type = ptr_type %.1 [symbolic = <unexpected>.inst+17 (constants.%.2)]
// CHECK:STDOUT:     %p.loc11_16.1: <unexpected>.inst+17 (%.2) = param p
// CHECK:STDOUT:     @F.%p: <unexpected>.inst+17 (%.2) = bind_name p, %p.loc11_16.1
// CHECK:STDOUT:     %T.ref.loc11_27: type = name_ref T, @F.%T [symbolic = <unexpected>.inst+15 (constants.%T)]
// CHECK:STDOUT:     %.loc11_28: type = ptr_type %T [symbolic = <unexpected>.inst+16 (constants.%.1)]
// CHECK:STDOUT:     @F.%return: ref %.1 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%T: type, %p: <unexpected>.inst+17 (%.2)) -> %.1
// CHECK:STDOUT:     generic [%T: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: <unexpected>.inst+17 (%.2) = name_ref p, %p
// CHECK:STDOUT:   %.loc12_10.1: ref <unexpected>.inst+16 (%.1) = deref %p.ref
// CHECK:STDOUT:   %.loc12_10.2: <unexpected>.inst+16 (%.1) = bind_value %.loc12_10.1
// CHECK:STDOUT:   return %.loc12_10.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%F.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   <unexpected>.inst+15 => constants.%T
// CHECK:STDOUT:   <unexpected>.inst+16 => constants.%.1
// CHECK:STDOUT:   <unexpected>.inst+17 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
