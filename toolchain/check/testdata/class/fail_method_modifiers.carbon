// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class FinalClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `abstract` not allowed on `fn` declaration in a non-abstract `class` definition.
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  abstract fn Abstract[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `virtual` not allowed on `fn` declaration in a non-abstract non-base `class` definition.
  // CHECK:STDERR:   virtual fn Virtual[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-13]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  virtual fn Virtual[self: Self]();
}

abstract class AbstractClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+3]]:3: ERROR: `default` not allowed on `fn` declaration outside of an interface.
  // CHECK:STDERR:   default fn Default[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  default fn Default[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+3]]:3: ERROR: `final` not allowed on `fn` declaration outside of an interface.
  // CHECK:STDERR:   final fn Final[self: Self]();
  // CHECK:STDERR:   ^~~~~
  final fn Final[self: Self]();
}

base class BaseClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `abstract` not allowed on `fn` declaration in a non-abstract `class` definition.
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: Containing definition here.
  // CHECK:STDERR: base class BaseClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  abstract fn Abstract[self: Self]();
}

// CHECK:STDOUT: --- fail_method_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FinalClass: type = class_type @FinalClass [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %AbstractClass: type = class_type @AbstractClass [template]
// CHECK:STDOUT:   %BaseClass: type = class_type @BaseClass [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace {.FinalClass = %FinalClass.decl, .AbstractClass = %AbstractClass.decl, .BaseClass = %BaseClass.decl} [template]
// CHECK:STDOUT:   %FinalClass.decl = class_decl @FinalClass, ()
// CHECK:STDOUT:   %AbstractClass.decl = class_decl @AbstractClass, ()
// CHECK:STDOUT:   %BaseClass.decl = class_decl @BaseClass, ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FinalClass {
// CHECK:STDOUT:   %Abstract: <function> = fn_decl @Abstract.1 [template]
// CHECK:STDOUT:   %Virtual: <function> = fn_decl @Virtual [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Abstract = %Abstract
// CHECK:STDOUT:   .Virtual = %Virtual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractClass {
// CHECK:STDOUT:   %Default: <function> = fn_decl @Default [template]
// CHECK:STDOUT:   %Final: <function> = fn_decl @Final [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Default = %Default
// CHECK:STDOUT:   .Final = %Final
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @BaseClass {
// CHECK:STDOUT:   %Abstract: <function> = fn_decl @Abstract.2 [template]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Abstract = %Abstract
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.1[%self: FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Virtual[%self: FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Default[%self: AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Final[%self: AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.2[%self: BaseClass]();
// CHECK:STDOUT:
