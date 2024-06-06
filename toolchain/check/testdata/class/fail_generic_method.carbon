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
// CHECK:STDOUT:   %Class.1: <type of Class> = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: <type of F> = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: <type of Int32> = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N 0 [symbolic]
// CHECK:STDOUT:   %.type: type = fn_type @.1 [template]
// CHECK:STDOUT:   %.4: <type of <invalid>> = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: <type of Class> = class_decl @Class [template = constants.%Class.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: <type of Int32> = import_ref ir3, inst+3, loaded [template = constants.%Int32]
// CHECK:STDOUT:   %.decl: <type of <invalid>> = fn_decl @.1 [template = constants.%.4] {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc32_14.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc32_14.2: type = converted %int.make_type_32, %.loc32_14.1 [template = i32]
// CHECK:STDOUT:     %N.loc32_10.1: i32 = param N
// CHECK:STDOUT:     %N.loc32_10.2: i32 = bind_symbolic_name N 0, %N.loc32_10.1 [symbolic = constants.%N]
// CHECK:STDOUT:     %Self.ref: <error> = name_ref Self, <error> [template = <error>]
// CHECK:STDOUT:     %self.loc32_21.1: <error> = param self
// CHECK:STDOUT:     @.1.%self: <error> = bind_name self, %self.loc32_21.1
// CHECK:STDOUT:     %T.ref: <error> = name_ref T, <error> [template = <error>]
// CHECK:STDOUT:     %n.loc32_33.1: <error> = param n
// CHECK:STDOUT:     @.1.%n: <error> = bind_name n, %n.loc32_33.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: <type of F> = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.2 [template = constants.%Class.2]
// CHECK:STDOUT:     %self.loc13_8.1: Class = param self
// CHECK:STDOUT:     %self.loc13_8.2: Class = bind_name self, %self.loc13_8.1
// CHECK:STDOUT:     %T.ref.loc13: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc13_20.1: T = param n
// CHECK:STDOUT:     %n.loc13_20.2: T = bind_name n, %n.loc13_20.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Class.2
// CHECK:STDOUT:   .a = %.loc12
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F[@Class.%self.loc13_8.2: Class](@Class.%n.loc13_20.2: T);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @.1[%self: <error>](%n: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
