// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class FinalClass {
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+9]]:3: ERROR: Semantics TODO: `private`.
  // CHECK:STDERR:   private abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:11: ERROR: `abstract` not allowed on `fn` declaration in a non-abstract `class` definition.
  // CHECK:STDERR:   private abstract fn Abstract[self: Self]();
  // CHECK:STDERR:           ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-7]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~
  private abstract fn Abstract[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+9]]:3: ERROR: Semantics TODO: `protected`.
  // CHECK:STDERR:   protected virtual fn Virtual[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:13: ERROR: `virtual` not allowed on `fn` declaration in a non-abstract non-base `class` definition.
  // CHECK:STDERR:   protected virtual fn Virtual[self: Self]();
  // CHECK:STDERR:             ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-17]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~
  protected virtual fn Virtual[self: Self]();
}

abstract class AbstractClass {
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `default` not allowed on `fn` declaration in a `class` definition.
  // CHECK:STDERR:   default fn Default[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-4]]:1: Containing definition here.
  // CHECK:STDERR: abstract class AbstractClass {
  // CHECK:STDERR: ^~~~~~~~
  default fn Default[self: Self]();
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `final` not allowed on `fn` declaration in a `class` definition.
  // CHECK:STDERR:   final fn Final[self: Self]();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-11]]:1: Containing definition here.
  // CHECK:STDERR: abstract class AbstractClass {
  // CHECK:STDERR: ^~~~~~~~
  final fn Final[self: Self]();
}

base class BaseClass {
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: ERROR: `abstract` not allowed on `fn` declaration in a non-abstract `class` definition.
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-4]]:1: Containing definition here.
  // CHECK:STDERR: base class BaseClass {
  // CHECK:STDERR: ^~~~
  abstract fn Abstract[self: Self]();
}

// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.loc28: type = struct_type {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file "fail_method_modifiers.carbon" {
// CHECK:STDOUT:   class_decl @FinalClass, ()
// CHECK:STDOUT:   %FinalClass: type = class_type @FinalClass
// CHECK:STDOUT:   class_decl @AbstractClass, ()
// CHECK:STDOUT:   %AbstractClass: type = class_type @AbstractClass
// CHECK:STDOUT:   class_decl @BaseClass, ()
// CHECK:STDOUT:   %BaseClass: type = class_type @BaseClass
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FinalClass {
// CHECK:STDOUT:   %Abstract: <function> = fn_decl @Abstract.1
// CHECK:STDOUT:   %Virtual: <function> = fn_decl @Virtual
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Abstract = %Abstract
// CHECK:STDOUT:   .Virtual = %Virtual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractClass {
// CHECK:STDOUT:   %Default: <function> = fn_decl @Default
// CHECK:STDOUT:   %Final: <function> = fn_decl @Final
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Default = %Default
// CHECK:STDOUT:   .Final = %Final
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @BaseClass {
// CHECK:STDOUT:   %Abstract: <function> = fn_decl @Abstract.2
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
