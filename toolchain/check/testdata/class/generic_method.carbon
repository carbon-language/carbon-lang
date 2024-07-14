// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic_method.carbon

class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

fn Class(T:! type).F[self: Self](n: T) {}

// CHECK:STDOUT: --- generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, file.%Class.decl(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %T} [symbolic]
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.loc16_10.1: type = param T
// CHECK:STDOUT:     %T.loc16_10.2: type = bind_symbolic_name T 0, %T.loc16_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %.loc16: type = specific_constant constants.%Class.2, %Class.decl(constants.%T) [symbolic = constants.%Class.2]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc16 [symbolic = constants.%Class.2]
// CHECK:STDOUT:     %self.loc16_22.1: %Class.2 = param self
// CHECK:STDOUT:     @F.%self: %Class.2 = bind_name self, %self.loc16_22.1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc16_34.1: %T = param n
// CHECK:STDOUT:     @F.%n: %T = bind_name n, %n.loc16_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc12 (constants.%T)]
// CHECK:STDOUT:   %.loc12: <unexpected instref inst+28> (%.2) = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %.loc13: type = specific_constant constants.%Class.2, file.%Class.decl(constants.%T) [symbolic = %.loc13 (constants.%Class.2)]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, %.loc13 [symbolic = %.loc13 (constants.%Class.2)]
// CHECK:STDOUT:     %self.loc13_8.1: @Class.%.loc13 (%Class.2) = param self
// CHECK:STDOUT:     %self.loc13_8.2: @Class.%.loc13 (%Class.2) = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, file.%T.loc11_13.2 [symbolic = %T.ref.loc13 (constants.%T)]
// CHECK:STDOUT:     %n.loc13_20.1: @Class.%T.ref.loc13 (%T) = param n
// CHECK:STDOUT:     %n.loc13_20.2: @Class.%T.ref.loc13 (%T) = bind_name n, %n.loc13_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[%self: %Class.2](%n: %T)
// CHECK:STDOUT:     generic [file.%T.loc11_13.2: type] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.%F.decl(constants.%T) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   @Class.%.loc13 => constants.%Class.2
// CHECK:STDOUT:   @Class.%T.ref.loc13 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific file.%Class.decl(@Class.%T.ref.loc12) {
// CHECK:STDOUT: declaration:
// CHECK:STDOUT:   file.%T.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
