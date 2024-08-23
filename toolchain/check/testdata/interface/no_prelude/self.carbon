// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/no_prelude/self.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/no_prelude/self.carbon

interface UseSelf {
  fn F[self: Self]() -> Self;
}

// CHECK:STDOUT: --- self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: type = interface_type @UseSelf [template]
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.2: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = assoc_entity_type %.1, %F.type [template]
// CHECK:STDOUT:   %.4: %.3 = assoc_entity element0, @UseSelf.%F.decl [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .UseSelf = %UseSelf.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %UseSelf.decl: type = interface_decl @UseSelf [template = constants.%.1] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @UseSelf {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = constants.%Self]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref.loc12_14: %.1 = name_ref Self, %Self [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_14.1: type = facet_type_access %Self.ref.loc12_14 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_14.2: type = converted %Self.ref.loc12_14, %.loc12_14.1 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %self.loc12_8.1: @F.%Self (%Self) = param self
// CHECK:STDOUT:     %self.loc12_8.2: @F.%Self (%Self) = bind_name self, %self.loc12_8.1
// CHECK:STDOUT:     %Self.ref.loc12_25: %.1 = name_ref Self, %Self [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_25.1: type = facet_type_access %Self.ref.loc12_25 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %.loc12_25.2: type = converted %Self.ref.loc12_25, %.loc12_25.1 [symbolic = @F.%Self (constants.%Self)]
// CHECK:STDOUT:     %return.var: ref @F.%Self (%Self) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_29: %.3 = assoc_entity element0, %F.decl [template = constants.%.4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   .F = %.loc12_29
// CHECK:STDOUT:   witness = (%F.decl)
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@UseSelf.%Self: %.1) {
// CHECK:STDOUT:   %Self: %.1 = bind_symbolic_name Self 0 [symbolic = %Self (constants.%Self)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[@UseSelf.%self.loc12_8.2: @F.%Self (%Self)]() -> @F.%Self (%Self);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%Self) {
// CHECK:STDOUT:   %Self => constants.%Self
// CHECK:STDOUT: }
// CHECK:STDOUT:
