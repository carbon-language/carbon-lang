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
// CHECK:STDOUT:     %T.loc11_6.1: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     @F.%T.loc11: type = bind_symbolic_name T 0, %T.loc11_6.1 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc11_19: type = name_ref T, @F.%T.loc11 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_20: type = ptr_type %T [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     %.loc11_21: type = ptr_type %.1 [symbolic = @F.%.2 (constants.%.2)]
// CHECK:STDOUT:     %p.loc11_16.1: @F.%.2 (%.2) = param p, runtime_param0
// CHECK:STDOUT:     @F.%p: @F.%.2 (%.2) = bind_name p, %p.loc11_16.1
// CHECK:STDOUT:     %T.ref.loc11_27: type = name_ref T, @F.%T.loc11 [symbolic = @F.%T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc11_28: type = ptr_type %T [symbolic = @F.%.1 (constants.%.1)]
// CHECK:STDOUT:     @F.%return: ref @F.%.1 (%.1) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   %.1: type = ptr_type @F.%T.1 (%T) [symbolic = %.1 (constants.%.1)]
// CHECK:STDOUT:   %.2: type = ptr_type @F.%.1 (%.1) [symbolic = %.2 (constants.%.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%T.loc11: type, %p: @F.%.2 (%.2)) -> @F.%.1 (%.1) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %p.ref: @F.%.2 (%.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc12_10.1: ref @F.%.1 (%.1) = deref %p.ref
// CHECK:STDOUT:     %.loc12_10.2: @F.%.1 (%.1) = bind_value %.loc12_10.1
// CHECK:STDOUT:     return %.loc12_10.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT:   %.1 => constants.%.1
// CHECK:STDOUT:   %.2 => constants.%.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
