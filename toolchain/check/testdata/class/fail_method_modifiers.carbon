// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

class FinalClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: ERROR: `abstract` not allowed on `fn` declaration in a non-abstract `class` definition.
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn Abstract[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: ERROR: `virtual` not allowed on `fn` declaration in a non-abstract non-base `class` definition.
  // CHECK:STDERR:   virtual fn Virtual[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-14]]:1: Containing definition here.
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn Virtual[self: Self]();
}

abstract class AbstractClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: ERROR: `default` not allowed on `fn` declaration outside of an interface.
  // CHECK:STDERR:   default fn Default[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn Default[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: ERROR: `final` not allowed on `fn` declaration outside of an interface.
  // CHECK:STDERR:   final fn Final[self: Self]();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
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
// CHECK:STDOUT:   %Abstract.1: type = fn_type @Abstract.1 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %struct.1: Abstract = struct_value () [template]
// CHECK:STDOUT:   %Virtual: type = fn_type @Virtual [template]
// CHECK:STDOUT:   %struct.2: Virtual = struct_value () [template]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %AbstractClass: type = class_type @AbstractClass [template]
// CHECK:STDOUT:   %Default: type = fn_type @Default [template]
// CHECK:STDOUT:   %struct.3: Default = struct_value () [template]
// CHECK:STDOUT:   %Final: type = fn_type @Final [template]
// CHECK:STDOUT:   %struct.4: Final = struct_value () [template]
// CHECK:STDOUT:   %BaseClass: type = class_type @BaseClass [template]
// CHECK:STDOUT:   %Abstract.2: type = fn_type @Abstract.2 [template]
// CHECK:STDOUT:   %struct.5: Abstract = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = %Core
// CHECK:STDOUT:     .FinalClass = %FinalClass.decl
// CHECK:STDOUT:     .AbstractClass = %AbstractClass.decl
// CHECK:STDOUT:     .BaseClass = %BaseClass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core: <namespace> = namespace [template] {}
// CHECK:STDOUT:   %FinalClass.decl: type = class_decl @FinalClass [template = constants.%FinalClass] {}
// CHECK:STDOUT:   %AbstractClass.decl: type = class_decl @AbstractClass [template = constants.%AbstractClass] {}
// CHECK:STDOUT:   %BaseClass.decl: type = class_decl @BaseClass [template = constants.%BaseClass] {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FinalClass {
// CHECK:STDOUT:   %Abstract.decl: Abstract = fn_decl @Abstract.1 [template = constants.%struct.1] {
// CHECK:STDOUT:     %Self.ref.loc16: type = name_ref Self, constants.%FinalClass [template = constants.%FinalClass]
// CHECK:STDOUT:     %self.loc16_24.1: FinalClass = param self
// CHECK:STDOUT:     %self.loc16_24.2: FinalClass = bind_name self, %self.loc16_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Virtual.decl: Virtual = fn_decl @Virtual [template = constants.%struct.2] {
// CHECK:STDOUT:     %Self.ref.loc25: type = name_ref Self, constants.%FinalClass [template = constants.%FinalClass]
// CHECK:STDOUT:     %self.loc25_22.1: FinalClass = param self
// CHECK:STDOUT:     %self.loc25_22.2: FinalClass = bind_name self, %self.loc25_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FinalClass
// CHECK:STDOUT:   .Abstract = %Abstract.decl
// CHECK:STDOUT:   .Virtual = %Virtual.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractClass {
// CHECK:STDOUT:   %Default.decl: Default = fn_decl @Default [template = constants.%struct.3] {
// CHECK:STDOUT:     %Self.ref.loc34: type = name_ref Self, constants.%AbstractClass [template = constants.%AbstractClass]
// CHECK:STDOUT:     %self.loc34_22.1: AbstractClass = param self
// CHECK:STDOUT:     %self.loc34_22.2: AbstractClass = bind_name self, %self.loc34_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Final.decl: Final = fn_decl @Final [template = constants.%struct.4] {
// CHECK:STDOUT:     %Self.ref.loc40: type = name_ref Self, constants.%AbstractClass [template = constants.%AbstractClass]
// CHECK:STDOUT:     %self.loc40_18.1: AbstractClass = param self
// CHECK:STDOUT:     %self.loc40_18.2: AbstractClass = bind_name self, %self.loc40_18.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractClass
// CHECK:STDOUT:   .Default = %Default.decl
// CHECK:STDOUT:   .Final = %Final.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @BaseClass {
// CHECK:STDOUT:   %Abstract.decl: Abstract = fn_decl @Abstract.2 [template = constants.%struct.5] {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%BaseClass [template = constants.%BaseClass]
// CHECK:STDOUT:     %self.loc51_24.1: BaseClass = param self
// CHECK:STDOUT:     %self.loc51_24.2: BaseClass = bind_name self, %self.loc51_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%BaseClass
// CHECK:STDOUT:   .Abstract = %Abstract.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.1[@FinalClass.%self.loc16_24.2: FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Virtual[@FinalClass.%self.loc25_22.2: FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Default[@AbstractClass.%self.loc34_22.2: AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Final[@AbstractClass.%self.loc40_18.2: AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.2[@BaseClass.%self.loc51_24.2: BaseClass]();
// CHECK:STDOUT:
