// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_method_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_method_modifiers.carbon

class FinalClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn Abstract[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+7]]:3: error: `virtual` not allowed; requires `abstract` or `base` class scope [ModifierVirtualNotAllowed]
  // CHECK:STDERR:   virtual fn Virtual[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-14]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: class FinalClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn Virtual[self: Self]();
}

abstract class AbstractClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: error: `default` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   default fn Default[self: Self]();
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  default fn Default[self: Self]();

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+4]]:3: error: `final` not allowed; requires interface scope [ModifierRequiresInterface]
  // CHECK:STDERR:   final fn Final[self: Self]();
  // CHECK:STDERR:   ^~~~~
  // CHECK:STDERR:
  final fn Final[self: Self]();
}

base class BaseClass {

  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE+6]]:3: error: `abstract` not allowed; requires `abstract` class scope [ModifierAbstractNotAllowed]
  // CHECK:STDERR:   abstract fn Abstract[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~
  // CHECK:STDERR: fail_method_modifiers.carbon:[[@LINE-5]]:1: note: containing definition here [ModifierNotInContext]
  // CHECK:STDERR: base class BaseClass {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~
  abstract fn Abstract[self: Self]();
}

// CHECK:STDOUT: --- fail_method_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %FinalClass: type = class_type @FinalClass [template]
// CHECK:STDOUT:   %Abstract.type.1: type = fn_type @Abstract.1 [template]
// CHECK:STDOUT:   %Abstract.1: %Abstract.type.1 = struct_value () [template]
// CHECK:STDOUT:   %Virtual.type: type = fn_type @Virtual [template]
// CHECK:STDOUT:   %Virtual: %Virtual.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %AbstractClass: type = class_type @AbstractClass [template]
// CHECK:STDOUT:   %Default.type: type = fn_type @Default [template]
// CHECK:STDOUT:   %Default: %Default.type = struct_value () [template]
// CHECK:STDOUT:   %Final.type: type = fn_type @Final [template]
// CHECK:STDOUT:   %Final: %Final.type = struct_value () [template]
// CHECK:STDOUT:   %BaseClass: type = class_type @BaseClass [template]
// CHECK:STDOUT:   %Abstract.type.2: type = fn_type @Abstract.2 [template]
// CHECK:STDOUT:   %Abstract.2: %Abstract.type.2 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .FinalClass = %FinalClass.decl
// CHECK:STDOUT:     .AbstractClass = %AbstractClass.decl
// CHECK:STDOUT:     .BaseClass = %BaseClass.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %FinalClass.decl: type = class_decl @FinalClass [template = constants.%FinalClass] {} {}
// CHECK:STDOUT:   %AbstractClass.decl: type = class_decl @AbstractClass [template = constants.%AbstractClass] {} {}
// CHECK:STDOUT:   %BaseClass.decl: type = class_decl @BaseClass [template = constants.%BaseClass] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @FinalClass {
// CHECK:STDOUT:   %Abstract.decl: %Abstract.type.1 = fn_decl @Abstract.1 [template = constants.%Abstract.1] {
// CHECK:STDOUT:     %self.patt: %FinalClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FinalClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FinalClass [template = constants.%FinalClass]
// CHECK:STDOUT:     %self.param: %FinalClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %FinalClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Virtual.decl: %Virtual.type = fn_decl @Virtual [template = constants.%Virtual] {
// CHECK:STDOUT:     %self.patt: %FinalClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %FinalClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%FinalClass [template = constants.%FinalClass]
// CHECK:STDOUT:     %self.param: %FinalClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %FinalClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%FinalClass
// CHECK:STDOUT:   .Abstract = %Abstract.decl
// CHECK:STDOUT:   .Virtual = %Virtual.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractClass {
// CHECK:STDOUT:   %Default.decl: %Default.type = fn_decl @Default [template = constants.%Default] {
// CHECK:STDOUT:     %self.patt: %AbstractClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractClass [template = constants.%AbstractClass]
// CHECK:STDOUT:     %self.param: %AbstractClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %AbstractClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Final.decl: %Final.type = fn_decl @Final [template = constants.%Final] {
// CHECK:STDOUT:     %self.patt: %AbstractClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractClass [template = constants.%AbstractClass]
// CHECK:STDOUT:     %self.param: %AbstractClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %AbstractClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc45: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractClass
// CHECK:STDOUT:   .Default = %Default.decl
// CHECK:STDOUT:   .Final = %Final.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @BaseClass {
// CHECK:STDOUT:   %Abstract.decl: %Abstract.type.2 = fn_decl @Abstract.2 [template = constants.%Abstract.2] {
// CHECK:STDOUT:     %self.patt: %BaseClass = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %BaseClass = value_param_pattern %self.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%BaseClass [template = constants.%BaseClass]
// CHECK:STDOUT:     %self.param: %BaseClass = value_param runtime_param0
// CHECK:STDOUT:     %self: %BaseClass = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc56: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%BaseClass
// CHECK:STDOUT:   .Abstract = %Abstract.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.1[%self.param_patt: %FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Virtual[%self.param_patt: %FinalClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Default[%self.param_patt: %AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Final[%self.param_patt: %AbstractClass]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Abstract.2[%self.param_patt: %BaseClass]();
// CHECK:STDOUT:
