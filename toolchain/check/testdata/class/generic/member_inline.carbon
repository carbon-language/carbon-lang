// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_inline.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_inline.carbon

class Class(T:! type) {
  fn F(n: T) -> T {
    return n;
  }

  fn G[self: Self]() -> T {
    return self.n;
  }

  var n: T;
}

// CHECK:STDOUT: --- member_inline.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, file.%Class.decl(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type %.3 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Core: <namespace> = namespace %Core.import, [template] {}
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.ref.loc12_11: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12_11 (constants.%T)]
// CHECK:STDOUT:     %n.loc12_8.1: @Class.%T.ref.loc12_11 (%T) = param n
// CHECK:STDOUT:     %n.loc12_8.2: @Class.%T.ref.loc12_11 (%T) = bind_name n, %n.loc12_8.1
// CHECK:STDOUT:     %T.ref.loc12_17: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12_11 (constants.%T)]
// CHECK:STDOUT:     %return.var.loc12: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %.loc16: type = specific_constant constants.%Class.2, file.%Class.decl(constants.%T) [symbolic = %.loc16 (constants.%Class.2)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc16 [symbolic = %.loc16 (constants.%Class.2)]
// CHECK:STDOUT:     %self.loc16_8.1: @Class.%.loc16 (%Class.2) = param self
// CHECK:STDOUT:     %self.loc16_8.2: @Class.%.loc16 (%Class.2) = bind_name self, %self.loc16_8.1
// CHECK:STDOUT:     %T.ref.loc16: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc16 (constants.%T)]
// CHECK:STDOUT:     %return.var.loc16: ref %T = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T.ref.loc20: type = name_ref T, file.%T.loc11_13.2 [symbolic = file.%T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %.loc20: <unexpected>.inst+35 (%.2) = field_decl n, element0 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   .n = %.loc20
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(@Class.%n.loc12_8.2: @Class.%T.ref.loc12_11 (%T)) -> %T
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %n.ref: @Class.%T.ref.loc12_11 (%T) = name_ref n, @Class.%n.loc12_8.2
// CHECK:STDOUT:   return %n.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G[@Class.%self.loc16_8.2: @Class.%.loc16 (%Class.2)]() -> %T
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %self.ref: @Class.%.loc16 (%Class.2) = name_ref self, @Class.%self.loc16_8.2
// CHECK:STDOUT:   %n.ref: <unexpected>.inst+44 (%.2) = name_ref n, @Class.%.loc20 [template = @Class.%.loc20]
// CHECK:STDOUT:   %.loc17_16.1: ref @Class.%T.ref.loc16 (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:   %.loc17_16.2: @Class.%T.ref.loc16 (%T) = bind_value %.loc17_16.1
// CHECK:STDOUT:   return %.loc17_16.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: definition:
// CHECK:STDOUT:   <unexpected>.inst+34 => constants.%Class.2
// CHECK:STDOUT:   <unexpected>.inst+35 => constants.%.2
// CHECK:STDOUT:   <unexpected>.inst+31.loc20_8 => <unexpected>.inst+32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.%F.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @Class.%T.ref.loc12_11 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.%G.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @Class.%.loc16 => constants.%Class.2
// CHECK:STDOUT:   @Class.%T.ref.loc16 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(file.%T.loc11_13.2) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
