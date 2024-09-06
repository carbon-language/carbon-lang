// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_todo_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_todo_modifiers.carbon

// TODO: Test calls to these (member access control is not yet implemented).
class Access {
  private fn F();

  protected fn G();

  private var k: i32;

  protected var l: i32;
}

base class Base {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn H();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn H();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn I();
  // CHECK:STDERR:   ^~~~
  // CHECK:STDERR:
  impl fn I();
}

abstract class Abstract {

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   abstract fn J();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR:
  abstract fn J();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+4]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   virtual fn K();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  virtual fn K();

  // CHECK:STDERR: fail_todo_modifiers.carbon:[[@LINE+3]]:3: ERROR: Semantics TODO: `method modifier`.
  // CHECK:STDERR:   impl fn L();
  // CHECK:STDERR:   ^~~~
  impl fn L();
}

// CHECK:STDOUT: --- fail_todo_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Access: type = class_type @Access [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Access, i32 [template]
// CHECK:STDOUT:   %.3: type = struct_type {.k: i32, .l: i32} [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = fn_type @I [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = struct_type {} [template]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %L.type: type = fn_type @L [template]
// CHECK:STDOUT:   %L: %L.type = struct_value () [template]
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
// CHECK:STDOUT:     .Access = %Access.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Access.decl: type = class_decl @Access [template = constants.%Access] {}
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {}
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Access {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {}
// CHECK:STDOUT:   %int.make_type_32.loc17: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc17_18.1: type = value_of_initializer %int.make_type_32.loc17 [template = i32]
// CHECK:STDOUT:   %.loc17_18.2: type = converted %int.make_type_32.loc17, %.loc17_18.1 [template = i32]
// CHECK:STDOUT:   %.loc17_16: %.2 = field_decl k, element0 [template]
// CHECK:STDOUT:   %int.make_type_32.loc19: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:   %.loc19_20.1: type = value_of_initializer %int.make_type_32.loc19 [template = i32]
// CHECK:STDOUT:   %.loc19_20.2: type = converted %int.make_type_32.loc19, %.loc19_20.1 [template = i32]
// CHECK:STDOUT:   %.loc19_18: %.2 = field_decl l, element1 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Access
// CHECK:STDOUT:   .F [private] = %F.decl
// CHECK:STDOUT:   .G [protected] = %G.decl
// CHECK:STDOUT:   .k [private] = %.loc17_16
// CHECK:STDOUT:   .l [protected] = %.loc19_18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {}
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [template = constants.%I] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {}
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [template = constants.%L] {}
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   .L = %L.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @L();
// CHECK:STDOUT:
