// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_todo_generic_method.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/check/testdata/class/fail_todo_generic_method.carbon

class Class(T:! type) {
  var a: T;
  fn F[self: Self](n: T);
}

// CHECK:STDERR: fail_todo_generic_method.carbon:[[@LINE+3]]:9: ERROR: Semantics TODO: `name qualifier with parameters`.
// CHECK:STDERR: fn Class(T:! type).F[self: Self](n: T) {}
// CHECK:STDERR:         ^~~~~~~~~~
fn Class(T:! type).F[self: Self](n: T) {}

// CHECK:STDOUT: --- fail_todo_generic_method.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T 0 [symbolic]
// CHECK:STDOUT:   %Class.1: type = generic_class_type @Class [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Class = struct_value () [template]
// CHECK:STDOUT:   %Class.2: type = class_type @Class [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type Class, T [symbolic]
// CHECK:STDOUT:   %F: type = fn_type @F [template]
// CHECK:STDOUT:   %struct.2: F = struct_value () [template]
// CHECK:STDOUT:   %.3: type = struct_type {.a: T} [symbolic]
// CHECK:STDOUT:   %.4: type = ptr_type {.a: T} [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %Class.decl: Class = class_decl @Class [template = constants.%struct.1] {
// CHECK:STDOUT:     %T.loc11_13.1: type = param T
// CHECK:STDOUT:     %T.loc11_13.2: type = bind_symbolic_name T 0, %T.loc11_13.1 [symbolic = constants.%T]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
// CHECK:STDOUT:     %T.loc19_10.1: type = param T
// CHECK:STDOUT:     %T.loc19_10.2: type = bind_symbolic_name T 0, %T.loc19_10.1 [symbolic = constants.%T]
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Class.2 [template = constants.%Class.2]
// CHECK:STDOUT:     %self.loc19_22.1: Class = param self
// CHECK:STDOUT:     @F.%self: Class = bind_name self, %self.loc19_22.1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_10.2 [symbolic = constants.%T]
// CHECK:STDOUT:     %n.loc19_34.1: T = param n
// CHECK:STDOUT:     @F.%n: T = bind_name n, %n.loc19_34.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Class {
// CHECK:STDOUT:   %T.ref.loc12: type = name_ref T, file.%T.loc11_13.2 [symbolic = constants.%T]
// CHECK:STDOUT:   %.loc12: <unbound element of class Class> = field_decl a, element0 [template]
// CHECK:STDOUT:   %F.decl: F = fn_decl @F [template = constants.%struct.2] {
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
// CHECK:STDOUT: fn @F[%self: Class](%n: T) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
