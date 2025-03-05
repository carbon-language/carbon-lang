// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/fail_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/fail_modifiers.carbon

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `private` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: private private class DuplicatePrivate;
// CHECK:STDERR:         ^~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private private class DuplicatePrivate;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private private class DuplicatePrivate;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `abstract` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: abstract class AbstractDecl;
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract class AbstractDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:9: error: `protected` not allowed on declaration with `private` [ModifierNotAllowedWith]
// CHECK:STDERR: private protected class TwoAccess;
// CHECK:STDERR:         ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `private` previously appeared here [ModifierPrevious]
// CHECK:STDERR: private protected class TwoAccess;
// CHECK:STDERR: ^~~~~~~
// CHECK:STDERR:
private protected class TwoAccess;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: error: `base` not allowed on `class` forward declaration, only definition [ModifierOnlyAllowedOnDefinition]
// CHECK:STDERR: base class BaseDecl;
// CHECK:STDERR: ^~~~
// CHECK:STDERR:
base class BaseDecl;

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `abstract` repeated on declaration [ModifierRepeated]
// CHECK:STDERR: abstract abstract class TwoAbstract { }
// CHECK:STDERR:          ^~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract abstract class TwoAbstract { }
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract abstract class TwoAbstract { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+15]]:19: error: `base` not allowed on declaration with `virtual` [ModifierNotAllowedWith]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:                   ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+12]]:11: note: `virtual` previously appeared here [ModifierPrevious]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+8]]:1: error: `protected` not allowed; requires class scope [ModifierProtectedNotAllowed]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR: ^~~~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:11: error: `virtual` not allowed on `class` declaration [ModifierNotAllowedOnDeclaration]
// CHECK:STDERR: protected virtual base class Virtual {}
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
protected virtual base class Virtual {}

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `protected` must appear before `abstract` [ModifierMustAppearBefore]
// CHECK:STDERR: abstract protected class WrongOrder { }
// CHECK:STDERR:          ^~~~~~~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract protected class WrongOrder { }
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract protected class WrongOrder { }

// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+7]]:10: error: `base` not allowed on declaration with `abstract` [ModifierNotAllowedWith]
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR:          ^~~~
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:1: note: `abstract` previously appeared here [ModifierPrevious]
// CHECK:STDERR: abstract base class AbstractAndBase {}
// CHECK:STDERR: ^~~~~~~~
// CHECK:STDERR:
abstract base class AbstractAndBase {}

abstract class AbstractWithDefinition {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:31: error: definition of `abstract` function [DefinedAbstractFunction]
  // CHECK:STDERR:   abstract fn F[self: Self]() {}
  // CHECK:STDERR:                               ^
  // CHECK:STDERR:
  abstract fn F[self: Self]() {}
  abstract fn G[self: Self]();
}
// CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:43: error: definition of `abstract` function [DefinedAbstractFunction]
// CHECK:STDERR: fn AbstractWithDefinition.G[self: Self]() {
// CHECK:STDERR:                                           ^
// CHECK:STDERR:
fn AbstractWithDefinition.G[self: Self]() {
}

// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DuplicatePrivate: type = class_type @DuplicatePrivate [concrete]
// CHECK:STDOUT:   %AbstractDecl: type = class_type @AbstractDecl [concrete]
// CHECK:STDOUT:   %TwoAccess: type = class_type @TwoAccess [concrete]
// CHECK:STDOUT:   %BaseDecl: type = class_type @BaseDecl [concrete]
// CHECK:STDOUT:   %TwoAbstract: type = class_type @TwoAbstract [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Virtual: type = class_type @Virtual [concrete]
// CHECK:STDOUT:   %WrongOrder: type = class_type @WrongOrder [concrete]
// CHECK:STDOUT:   %AbstractAndBase: type = class_type @AbstractAndBase [concrete]
// CHECK:STDOUT:   %AbstractWithDefinition: type = class_type @AbstractWithDefinition [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.9a5: <vtable> = vtable (%F, %G) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .DuplicatePrivate [private] = %DuplicatePrivate.decl
// CHECK:STDOUT:     .AbstractDecl = %AbstractDecl.decl
// CHECK:STDOUT:     .TwoAccess [private] = %TwoAccess.decl
// CHECK:STDOUT:     .BaseDecl = %BaseDecl.decl
// CHECK:STDOUT:     .TwoAbstract = %TwoAbstract.decl
// CHECK:STDOUT:     .Virtual = %Virtual.decl
// CHECK:STDOUT:     .WrongOrder = %WrongOrder.decl
// CHECK:STDOUT:     .AbstractAndBase = %AbstractAndBase.decl
// CHECK:STDOUT:     .AbstractWithDefinition = %AbstractWithDefinition.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %DuplicatePrivate.decl: type = class_decl @DuplicatePrivate [concrete = constants.%DuplicatePrivate] {} {}
// CHECK:STDOUT:   %AbstractDecl.decl: type = class_decl @AbstractDecl [concrete = constants.%AbstractDecl] {} {}
// CHECK:STDOUT:   %TwoAccess.decl: type = class_decl @TwoAccess [concrete = constants.%TwoAccess] {} {}
// CHECK:STDOUT:   %BaseDecl.decl: type = class_decl @BaseDecl [concrete = constants.%BaseDecl] {} {}
// CHECK:STDOUT:   %TwoAbstract.decl: type = class_decl @TwoAbstract [concrete = constants.%TwoAbstract] {} {}
// CHECK:STDOUT:   %Virtual.decl: type = class_decl @Virtual [concrete = constants.%Virtual] {} {}
// CHECK:STDOUT:   %WrongOrder.decl: type = class_decl @WrongOrder [concrete = constants.%WrongOrder] {} {}
// CHECK:STDOUT:   %AbstractAndBase.decl: type = class_decl @AbstractAndBase [concrete = constants.%AbstractAndBase] {} {}
// CHECK:STDOUT:   %AbstractWithDefinition.decl: type = class_decl @AbstractWithDefinition [concrete = constants.%AbstractWithDefinition] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: %AbstractWithDefinition = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractWithDefinition = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc97: %AbstractWithDefinition = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc97: type = name_ref Self, constants.%AbstractWithDefinition [concrete = constants.%AbstractWithDefinition]
// CHECK:STDOUT:     %self.loc97: %AbstractWithDefinition = bind_name self, %self.param.loc97
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @DuplicatePrivate;
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractDecl;
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAccess;
// CHECK:STDOUT:
// CHECK:STDOUT: class @BaseDecl;
// CHECK:STDOUT:
// CHECK:STDOUT: class @TwoAbstract {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%TwoAbstract
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Virtual {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Virtual
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @WrongOrder {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%WrongOrder
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractAndBase {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractAndBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractWithDefinition {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %AbstractWithDefinition = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractWithDefinition = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AbstractWithDefinition = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractWithDefinition [concrete = constants.%AbstractWithDefinition]
// CHECK:STDOUT:     %self: %AbstractWithDefinition = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %self.patt: %AbstractWithDefinition = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractWithDefinition = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param.loc91: %AbstractWithDefinition = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc91: type = name_ref Self, constants.%AbstractWithDefinition [concrete = constants.%AbstractWithDefinition]
// CHECK:STDOUT:     %self.loc91: %AbstractWithDefinition = bind_name self, %self.param.loc91
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc92: <vtable> = vtable (%F.decl, %G.decl) [concrete = constants.%.9a5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractWithDefinition
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @F[%self.param_patt: %AbstractWithDefinition]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @G[%self.param_patt: %AbstractWithDefinition]() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
