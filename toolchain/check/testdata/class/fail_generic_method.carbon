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
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+14]]:10: ERROR: Redeclaration differs at parameter 1.
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:          ^
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE-10]]:13: Previous declaration's corresponding parameter here.
// CHECK:STDERR: class Class(T:! type) {
// CHECK:STDERR:             ^
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+7]]:27: ERROR: Name `Self` not found.
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:                           ^~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_generic_method.carbon:[[@LINE+3]]:36: ERROR: Name `T` not found.
// CHECK:STDERR: fn Class(N:! i32).F[self: Self](n: T) {}
// CHECK:STDERR:                                    ^
fn Class(N:! i32).F[self: Self](n: T) {}

// CHECK:STDOUT: --- fail_generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Class.1: %Class.type = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Class.2, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %.3: type = struct_type {.a: %T} [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: %.type = struct_value () [template]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T 0, %T.param [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.decl: %.type = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %self.patt: <error> = binding_pattern self
// CHECK:STDOUT:     %n.patt: <error> = binding_pattern n
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc32_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc32_14.2: type = converted %int.make_type_32, %.loc32_14.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N
// CHECK:STDOUT:     %N.loc32: i32 = bind_symbolic_name N 0, %N.param [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %self.param: <error> = param self
// CHECK:STDOUT:     %self: <error> = bind_name self, %self.param
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %n.param: <error> = param n
// CHECK:STDOUT:     %n: <error> = bind_name n, %n.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc11: type) {
// CHECK:STDOUT:   %T.1: type = bind_symbolic_name T 0 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.1) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:   %.1: type = unbound_element_type @Class.%Class (%Class.2), @Class.%T.1 (%T) [symbolic = %.1 (constants.%.2)]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.1) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc11 [symbolic = %T.1 (constants.%T)]
// CHECK:STDOUT:     %.loc12: @Class.%.1 (%.2) = field_decl a, element0 [template]
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%Class (%Class.2) = binding_pattern self
// CHECK:STDOUT:       %n.patt: @F.%T (%T) = binding_pattern n
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc13: type = specific_constant constants.%Class.2, @Class(constants.%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, %.loc13 [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:       %self.param: @F.%Class (%Class.2) = param self
// CHECK:STDOUT:       %self: @F.%Class (%Class.2) = bind_name self, %self.param
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Class.%T.loc11 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %n.param: @F.%T (%T) = param n
// CHECK:STDOUT:       %n: @F.%T (%T) = bind_name n, %n.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class.2
// CHECK:STDOUT:     .a = %.loc12
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc11: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic = %Class (constants.%Class.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: @F.%Class (%Class.2)](%n: @F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @.1(%N.loc32: i32) {
// CHECK:STDOUT:   %N.1: i32 = bind_symbolic_name N 0 [symbolic = %N.1 (constants.%N)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self: <error>](%n: <error>) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@F.%T) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@Class.%T.1) {
// CHECK:STDOUT:   %T.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @.1(constants.%N) {
// CHECK:STDOUT:   %N.1 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
