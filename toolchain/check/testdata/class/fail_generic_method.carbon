// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_generic_method.carbon

class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

// TODO: The follow-on errors here aren't great. Investigate whether we can
// enter the scope anyway if the parameters don't match.
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+14]]:10: error: type `i32` of parameter 1 in redeclaration differs from previous parameter type `type` [RedeclParamDiffersType]
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:          ^~~~~~~
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE-10]]:13: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR:             ^~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+7]]:27: error: name `Self` not found [NameNotFound]
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+3]]:36: error: name `T` not found [NameNotFound]
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:                                    ^
fn Class(N:! i32).F[self: Self](n: T) {}

// CHECK:STDOUT: --- fail_generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [symbolic]
// CHECK:STDOUT:   %.4: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.4 [template]
// CHECK:STDOUT:   %N: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.5: %.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt.loc11_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc11_13.1, runtime_param<invalid> [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc11_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.5] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: <error> = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:     %n.patt: <error> = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: <error> = value_param_pattern %n.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc32_14.1: Core.IntLiteral = int_value 32 [template = constants.%.4]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc32_14.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_14.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc32_14.3: type = converted %int.make_type_signed, %.loc32_14.2 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc32_10.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc32_10.2 (constants.%N)]
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: <error> = value_param runtime_param0
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %n.param: <error> = value_param runtime_param1
// CHECK:STDOUT:     %n: <error> = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T.loc11_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc11_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc11_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc11_13.2) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.loc12_8.2: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.loc11_13.2 (%T) [symbolic = %.loc12_8.2 (constants.%.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc11_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %.loc14_1.2: type = struct_type {.a: @Class.%T.loc11_13.2 (%T)} [symbolic = %.loc14_1.2 (constants.%.2)]
// CHECK:STDOUT:   %.loc14_1.3: <witness> = complete_type_witness @Class.%.loc14_1.2 (%.2) [symbolic = %.loc14_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11_13.1 [symbolic = %T.loc11_13.2 (constants.%T)]
// CHECK:STDOUT:     %.loc12_8.1: @Class.%.loc12_8.2 (%.1) = field_decl a, element0 [template]
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Class (%Class.2) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.%Class (%Class.2) = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:       %n.patt: @F.%T (%T) = binding_pattern n
// CHECK:STDOUT:       %n.param_patt: @F.%T (%T) = value_param_pattern %n.patt, runtime_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc13 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %self.param: @F.%Class (%Class.2) = value_param runtime_param0
// CHECK:STDOUT:       %self: @F.%Class (%Class.2) = bind_name self, %self.param
// CHECK:STDOUT:       %n.param: @F.%T (%T) = value_param runtime_param1
// CHECK:STDOUT:       %n: @F.%T (%T) = bind_name n, %n.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc14_1.1: <witness> = complete_type_witness %.2 [symbolic = %.loc14_1.3 (constants.%.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .a = %.loc12_8.1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc11_13.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%Class (%Class.2)](%n.param_patt: @F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%N.loc32_10.1: %i32) {
// CHECK:STDOUT:   %N.loc32_10.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc32_10.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: <error>](%n.param_patt: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc11_13.2) {
// CHECK:STDOUT:   %T.loc11_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc11_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%N) {
// CHECK:STDOUT:   %N.loc32_10.2 => constants.%N
// CHECK:STDOUT:   %N.patt => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
