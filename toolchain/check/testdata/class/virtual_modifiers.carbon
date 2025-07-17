// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/virtual_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/virtual_modifiers.carbon

// --- modifiers.carbon

package Modifiers;

base class Base {
  virtual fn H[self: Self]();
}

abstract class Abstract {
  abstract fn J[self: Self]();

  virtual fn K[self: Self]();
}

// --- override_import.carbon

library "[[@TEST_NAME]]";

import Modifiers;

class Derived {
  extend base: Modifiers.Base;
  impl fn H[self: Self]();
}

fn Use() {
  var d: Derived = {.base = {}};
}

// --- todo_fail_later_base.carbon

library "[[@TEST_NAME]]";

import Modifiers;

base class Derived {
  virtual fn F[self: Self]();
  extend base: Modifiers.Base;
}

// --- init.carbon

library "[[@TEST_NAME]]";

import Modifiers;

fn F() {
  var v: Modifiers.Base = {};
}

// --- impl_abstract.carbon

library "[[@TEST_NAME]]";

abstract class A1 {
  virtual fn F[self: Self]();
}

abstract class A2 {
  extend base: A1;
  impl fn F[self: Self]();
}

// --- impl_base.carbon

library "[[@TEST_NAME]]";

base class B1 {
  virtual fn F[self: Self]();
}

base class B2 {
  extend base: B1;
  impl fn F[self: Self]();
}

class C {
  extend base: B2;
  impl fn F[self: Self]();
}

fn Use() {
  var b1: B1 = {};
  var b2: B2 = {.base = {}};
  var c: C = {.base = {.base = {}}};
}

// --- fail_modifiers.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:3: error: impl without base class [ImplWithoutBase]
  // CHECK:STDERR:   impl fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F[self: Self]();
}

// --- init_members.carbon

library "[[@TEST_NAME]]";

base class Base {
  var m1: i32;
  var m2: i32;

  virtual fn F[self: Self]();
}

fn F() {
  var i: i32 = 3;
  // TODO: These should initialize element1 (.m), not element0 (the vptr)
  var b1: Base = {.m2 = i, .m1 = i};
  var b2: Base = {.m2 = 3, .m1 = 5};

  // This one is good, though.
  b1.m2 = 4;
}

// --- todo_fail_impl_without_base_declaration.carbon

library "[[@TEST_NAME]]";

base class Base {
}

class Derived {
  extend base: Base;
  impl fn F[self: Self]();
}

// --- abstract_impl.carbon

library "[[@TEST_NAME]]";

abstract class AbstractBase {
  abstract fn F[self: Self]();
}

abstract class AbstractIntermediate {
  extend base: AbstractBase;
}

class Derived {
  extend base: AbstractIntermediate;
  impl fn F[self: Self]();
}

// --- virtual_impl.carbon

library "[[@TEST_NAME]]";

base class VirtualBase {
  virtual fn F[self: Self]();
}

base class VirtualIntermediate {
  extend base: VirtualBase;
}

class Derived {
  extend base: VirtualIntermediate;
  impl fn F[self: Self]();
}

// --- fail_impl_mismatch.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE+7]]:3: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
  // CHECK:STDERR:   impl fn F[self: Self](v: i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE-8]]:3: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F[self: Self](v: i32);
}

// --- fail_todo_impl_conversion.carbon

library "[[@TEST_NAME]]";

class T1 {
}

class T2 {
}

impl T2 as Core.ImplicitAs(T1) {
  fn Convert[self: Self]() -> T1 {
    return {};
  }
}

base class Base {
  virtual fn F[self: Self]() -> T1;
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_todo_impl_conversion.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `T2` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   impl fn F[self: Self]() -> T2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_conversion.carbon:[[@LINE-8]]:3: note: previously declared with return type `T1` [FunctionRedeclReturnTypePrevious]
  // CHECK:STDERR:   virtual fn F[self: Self]() -> T1;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F[self: Self]() -> T2;
}

// --- fail_generic_virtual_decl.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_generic_virtual_decl.carbon:[[@LINE+3]]:1: error: use of undefined generic function [MissingGenericFunctionDefinition]
// CHECK:STDERR: base class Base(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~
base class Base(T:! type) {
  // CHECK:STDERR: fail_generic_virtual_decl.carbon:[[@LINE+4]]:3: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn F[self: Self]();
}


// --- impl_generic_base.carbon

library "[[@TEST_NAME]]";

class T1 {
}

base class Base(T:! type) {
  virtual fn F[self: Self](t: T) { }
}

class Derived {
  extend base: Base(T1);
  impl fn F[self: Self](t: T1) { }
}

// --- fail_virtual_without_self.carbon

library "[[@TEST_NAME]]";

abstract class T1 {
  // CHECK:STDERR: fail_virtual_without_self.carbon:[[@LINE+4]]:3: error: virtual class function [VirtualWithoutSelf]
  // CHECK:STDERR:   virtual fn F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn F();
  // CHECK:STDERR: fail_virtual_without_self.carbon:[[@LINE+4]]:3: error: virtual class function [VirtualWithoutSelf]
  // CHECK:STDERR:   abstract fn G();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn G();
}

class T2 {
  extend base: T1;
  // CHECK:STDERR: fail_virtual_without_self.carbon:[[@LINE+4]]:3: error: virtual class function [VirtualWithoutSelf]
  // CHECK:STDERR:   impl fn F();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F();
}

// --- fail_addr_self_mismatch.carbon

library "[[@TEST_NAME]]";

base class T1 {
  virtual fn F1[self: Self*]();
}

class T2 {
  extend base: T1;
  // CHECK:STDERR: fail_addr_self_mismatch.carbon:[[@LINE+7]]:14: error: redeclaration differs at implicit parameter 1 [RedeclParamDiffers]
  // CHECK:STDERR:   impl fn F1[addr self: Self*]();
  // CHECK:STDERR:              ^~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_addr_self_mismatch.carbon:[[@LINE-8]]:17: note: previous declaration's corresponding implicit parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F1[self: Self*]();
  // CHECK:STDERR:                 ^~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F1[addr self: Self*]();
}

// --- fail_generic_virtual.carbon

library "[[@TEST_NAME]]";

base class T1 {
  // CHECK:STDERR: fail_generic_virtual.carbon:[[@LINE+4]]:3: error: generic virtual function [GenericVirtual]
  // CHECK:STDERR:   virtual fn F[self: Self, T:! type]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn F[self: Self, T:! type]();
}

// --- fail_generic_virtual_in_generic_class.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  // CHECK:STDERR: fail_generic_virtual_in_generic_class.carbon:[[@LINE+4]]:3: error: generic virtual function [GenericVirtual]
  // CHECK:STDERR:   virtual fn F[self: Self, T:! type]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  virtual fn F[self: Self, T:! type]();
}

// --- generic_with_virtual.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  virtual fn F[self: Self]() { }
}

// --- with_dependent_arg.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  virtual fn F[self: Self](t: T) { }
}

// --- vtable_import_unneeded.carbon

library "[[@TEST_NAME]]";

import Modifiers;

fn F(b: Modifiers.Base);

// --- generic_derived_from_nongeneric.carbon

library "[[@TEST_NAME]]";

base class NonGenericBase {
  virtual fn F1[self: Self]() { }
  virtual fn F2[self: Self]() { }
}

base class GenericDerived(T:! type) {
  extend base: NonGenericBase;
  impl fn F2[self: Self]() { }
  virtual fn F3[self: Self]() { }
}

// --- nongeneric_derived_from_generic.carbon

library "[[@TEST_NAME]]";

base class GenericBase(T:! type) {
  virtual fn F1[self: Self]() { }
  virtual fn F2[self: Self]() { }
}

class T1;

base class NonGenericDerived {
  extend base: GenericBase(T1);
  impl fn F2[self: Self]() { }
  virtual fn F3[self: Self]() { }
}

// --- impl_generic_specifically.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[self: Self](t: Base(T)*) { }
}
class T1;
class D1 {
  extend base: Base(T1);
  impl fn F[self: Self](t: Base(T1)*) { }
}

// --- fail_impl_generic_specifically_mismatch.carbon

library "[[@TEST_NAME]]";

class T1;
class T2;

base class Base(T:! type) {
  virtual fn F[self: Self](t: T1*) { }
}

class D1 {
  extend base: Base(T1);
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE+7]]:25: error: type `<pattern for T2*>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T1*>` [RedeclParamDiffersType]
  // CHECK:STDERR:   impl fn F[self: Self](t: T2*) { }
  // CHECK:STDERR:                         ^~~~~~
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE-8]]:28: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self](t: T1*) { }
  // CHECK:STDERR:                            ^~~~~~
  // CHECK:STDERR:
  impl fn F[self: Self](t: T2*) { }
}

// --- fail_impl_generic_generic_mismatch.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  virtual fn F[self: Self](t: T*) { }
}
class Derived(T:! type) {
  extend base: Base(T);
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE+7]]:25: error: type `<pattern for T>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T*>` [RedeclParamDiffersType]
  // CHECK:STDERR:   impl fn F[self: Self](t: T) { }
  // CHECK:STDERR:                         ^~~~
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE-7]]:28: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self](t: T*) { }
  // CHECK:STDERR:                            ^~~~~
  // CHECK:STDERR:
  impl fn F[self: Self](t: T) { }
}

// --- impl_generic_generic.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  virtual fn F[self: Self](t: T) { }
}
class Derived(T:! type) {
  extend base: Base(T*);
  impl fn F[self: Self](t: T*) { }
}

// --- fail_todo_abstract_generic_undefined.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_abstract_generic_undefined.carbon:[[@LINE+3]]:1: error: use of undefined generic function [MissingGenericFunctionDefinition]
// CHECK:STDERR: abstract class Base(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
abstract class Base(T:! type) {
  // CHECK:STDERR: fail_todo_abstract_generic_undefined.carbon:[[@LINE+11]]:3: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:   abstract fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_abstract_generic_undefined.carbon:[[@LINE-5]]:1: error: use of undefined generic function [MissingGenericFunctionDefinition]
  // CHECK:STDERR: abstract class Base(T:! type) {
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_abstract_generic_undefined.carbon:[[@LINE+4]]:3: note: generic function declared here [MissingGenericFunctionDefinitionHere]
  // CHECK:STDERR:   abstract fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  abstract fn F[self: Self]();
}

class T1;

class Derived {
  extend base: Base(T1);
  impl fn F[self: Self]() { }
}

// --- generic_lib.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[self: Self]() { }
}

// --- generic_import.carbon

library "[[@TEST_NAME]]";

import library "generic_lib";

class T1;
var v: Base(T1) = {};

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.41e: type = pattern_type %Abstract [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract.vtable_ptr: ref %ptr = vtable_ptr @Abstract.vtable [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [concrete = constants.%Abstract] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [concrete = constants.%J] {
// CHECK:STDOUT:     %self.patt: %pattern_type.41e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.41e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [concrete = constants.%K] {
// CHECK:STDOUT:     %self.patt: %pattern_type.41e = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.41e = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr = vtable_ptr @Abstract.vtable [concrete = constants.%Abstract.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Abstract.vtable {
// CHECK:STDOUT:   @Abstract.%J.decl
// CHECK:STDOUT:   @Abstract.%K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @H(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @J(%self.param: %Abstract);
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @K(%self.param: %Abstract);
// CHECK:STDOUT:
// CHECK:STDOUT: --- override_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %H.type.dba: type = fn_type @H.loc8 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %H.bce: %H.type.dba = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.09e: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %H.c1b: %H.type.09e = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.96c: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base.96c [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Derived.vtable_ptr) [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.102: type = fn_type @Op.2, @Destroy.impl(%Derived) [concrete]
// CHECK:STDOUT:   %Op.539: %Op.type.102 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.404: type = ptr_type %Derived [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.539, @Op.2(%Derived) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Modifiers.import_ref.be7: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   %Modifiers.import_ref.1f3 = import_ref Modifiers//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.2cc = import_ref Modifiers//default, loc5_29, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc7: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %H.decl: %H.type.dba = fn_decl @H.loc8 [concrete = constants.%H.bce] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base.96c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Modifiers = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_ptr = imports.%Modifiers.import_ref.be7
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%H.c1b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @H.loc8(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.1 [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.fb9 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %Derived = var %d.var_patt
// CHECK:STDOUT:   %.loc12_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_31.1: %struct_type.base.f5e = struct_literal (%.loc12_30.1)
// CHECK:STDOUT:   %.loc12_31.2: ref %Base = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc12_30.2: ref %ptr.454 = class_element_access %.loc12_31.2, element0
// CHECK:STDOUT:   %.loc12_30.3: init %ptr.454 = initialize_from @Derived.%vtable_ptr to %.loc12_30.2 [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %.loc12_30.4: init %Base = class_init (%.loc12_30.3), %.loc12_31.2 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc12_31.3: init %Base = converted %.loc12_30.1, %.loc12_30.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc12_31.4: init %Derived = class_init (%.loc12_31.3), %d.var [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %.loc12_3: init %Derived = converted %.loc12_31.1, %.loc12_31.4 [concrete = constants.%Derived.val]
// CHECK:STDOUT:   assign %d.var, %.loc12_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d: ref %Derived = bind_name d, %d.var
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %d.var, constants.%Op.539
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.539, @Op.2(constants.%Derived) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %d.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.404 = addr_of %d.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Modifiers.import_ref.be7: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   %Modifiers.import_ref.1f3 = import_ref Modifiers//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.2cc = import_ref Modifiers//default, loc5_29, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .Modifiers = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_ptr = imports.%Modifiers.import_ref.be7
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%H
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.47e: type = fn_type @Op.2, @Destroy.impl(%Base) [concrete]
// CHECK:STDOUT:   %Op.033: %Op.type.47e = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f03: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.033, @Op.2(%Base) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Modifiers.import_ref.be7: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   %Modifiers.import_ref.1f3 = import_ref Modifiers//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.2cc = import_ref Modifiers//default, loc5_29, unloaded
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_ptr = imports.%Modifiers.import_ref.be7
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.80f = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.80f = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base = var %v.var_patt
// CHECK:STDOUT:   %.loc7_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_28.2: ref %ptr.454 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc7_28.3: init %ptr.454 = initialize_from imports.%Modifiers.import_ref.be7 to %.loc7_28.2 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc7_28.4: init %Base = class_init (%.loc7_28.3), %v.var [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_3: init %Base = converted %.loc7_28.1, %.loc7_28.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_19: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   %Op.bound: <bound method> = bound_method %v.var, constants.%Op.033
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function constants.%Op.033, @Op.2(constants.%Base) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %v.var, %Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f03 = addr_of %v.var
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.ddb: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %F.type.13a: type = fn_type @F.loc5 [concrete]
// CHECK:STDOUT:   %F.df5: %F.type.13a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %A1.vtable_ptr: ref %ptr.454 = vtable_ptr @A1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %A1 [concrete]
// CHECK:STDOUT:   %pattern_type.b21: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %F.type.4ae: type = fn_type @F.loc10 [concrete]
// CHECK:STDOUT:   %F.1d5: %F.type.4ae = struct_value () [concrete]
// CHECK:STDOUT:   %A2.vtable_ptr: ref %ptr.454 = vtable_ptr @A2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A1} [concrete]
// CHECK:STDOUT:   %complete_type.a6f: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [concrete = constants.%A1] {} {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [concrete = constants.%A2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %F.decl: %F.type.13a = fn_decl @F.loc5 [concrete = constants.%F.df5] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ddb = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ddb = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A1 [concrete = constants.%A1]
// CHECK:STDOUT:     %self: %A1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @A1.vtable [concrete = constants.%A1.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %A1.ref: type = name_ref A1, file.%A1.decl [concrete = constants.%A1]
// CHECK:STDOUT:   %.loc9: %A2.elem = base_decl %A1.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.4ae = fn_decl @F.loc10 [concrete = constants.%F.1d5] {
// CHECK:STDOUT:     %self.patt: %pattern_type.b21 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b21 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A2 [concrete = constants.%A2]
// CHECK:STDOUT:     %self: %A2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @A2.vtable [concrete = constants.%A2.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A1} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.a6f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .A1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @A1.vtable {
// CHECK:STDOUT:   @A1.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @A2.vtable {
// CHECK:STDOUT:   @A2.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc5(%self.param: %A1);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc10(%self.param: %A2);
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.e47: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %F.type.e4c: type = fn_type @F.loc5 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.8f5: %F.type.e4c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %B1.vtable_ptr: ref %ptr.454 = vtable_ptr @B1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %B2.elem: type = unbound_element_type %B2, %B1 [concrete]
// CHECK:STDOUT:   %pattern_type.b19: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %F.type.b26: type = fn_type @F.loc10 [concrete]
// CHECK:STDOUT:   %F.d48: %F.type.b26 = struct_value () [concrete]
// CHECK:STDOUT:   %B2.vtable_ptr: ref %ptr.454 = vtable_ptr @B2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.508: type = struct_type {.base: %B1} [concrete]
// CHECK:STDOUT:   %complete_type.5ac: <witness> = complete_type_witness %struct_type.base.508 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B2 [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %C.vtable_ptr: ref %ptr.454 = vtable_ptr @C.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.421: type = struct_type {.base: %B2} [concrete]
// CHECK:STDOUT:   %complete_type.066: <witness> = complete_type_witness %struct_type.base.421 [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %B1.val.267: %B1 = struct_value (%B1.vtable_ptr) [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %B1.val.d2f: %B1 = struct_value (%B2.vtable_ptr) [concrete]
// CHECK:STDOUT:   %B2.val.e9e: %B2 = struct_value (%B1.val.d2f) [concrete]
// CHECK:STDOUT:   %struct_type.base.a0c: type = struct_type {.base: %struct_type.base.f5e} [concrete]
// CHECK:STDOUT:   %B1.val.b9d: %B1 = struct_value (%C.vtable_ptr) [concrete]
// CHECK:STDOUT:   %B2.val.426: %B2 = struct_value (%B1.val.b9d) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B2.val.426) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.2, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6eb: <specific function> = specific_function %Op.5d7, @Op.2(%C) [concrete]
// CHECK:STDOUT:   %Op.type.49c: type = fn_type @Op.2, @Destroy.impl(%B2) [concrete]
// CHECK:STDOUT:   %Op.bb0: %Op.type.49c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.afe: type = ptr_type %B2 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.a6f: <specific function> = specific_function %Op.bb0, @Op.2(%B2) [concrete]
// CHECK:STDOUT:   %Op.type.5ba: type = fn_type @Op.2, @Destroy.impl(%B1) [concrete]
// CHECK:STDOUT:   %Op.a5c: %Op.type.5ba = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.890: type = ptr_type %B1 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.075: <specific function> = specific_function %Op.a5c, @Op.2(%B1) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .B1 = %B1.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .Use = %Use.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B1.decl: type = class_decl @B1 [concrete = constants.%B1] {} {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [concrete = constants.%B2] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %Use.decl: %Use.type = fn_decl @Use [concrete = constants.%Use] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B1 {
// CHECK:STDOUT:   %F.decl: %F.type.e4c = fn_decl @F.loc5 [concrete = constants.%F.8f5] {
// CHECK:STDOUT:     %self.patt: %pattern_type.e47 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.e47 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B1 [concrete = constants.%B1]
// CHECK:STDOUT:     %self: %B1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @B1.vtable [concrete = constants.%B1.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %.loc9: %B2.elem = base_decl %B1.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.b26 = fn_decl @F.loc10 [concrete = constants.%F.d48] {
// CHECK:STDOUT:     %self.patt: %pattern_type.b19 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b19 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B2 [concrete = constants.%B2]
// CHECK:STDOUT:     %self: %B2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @B2.vtable [concrete = constants.%B2.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B1} [concrete = constants.%struct_type.base.508]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.5ac]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .B1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B2.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.c29 = fn_decl @F.loc15 [concrete = constants.%F.437] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @C.vtable [concrete = constants.%C.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %B2} [concrete = constants.%struct_type.base.421]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.066]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B2 = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @B1.vtable {
// CHECK:STDOUT:   @B1.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @B2.vtable {
// CHECK:STDOUT:   @B2.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @C.vtable {
// CHECK:STDOUT:   @C.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc5(%self.param: %B1);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc10(%self.param: %B2);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc15(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.e47 = binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.e47 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %B1 = var %b1.var_patt
// CHECK:STDOUT:   %.loc19_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_17.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc19_17.3: init %ptr.454 = initialize_from @B1.%vtable_ptr to %.loc19_17.2 [concrete = constants.%B1.vtable_ptr]
// CHECK:STDOUT:   %.loc19_17.4: init %B1 = class_init (%.loc19_17.3), %b1.var [concrete = constants.%B1.val.267]
// CHECK:STDOUT:   %.loc19_3: init %B1 = converted %.loc19_17.1, %.loc19_17.4 [concrete = constants.%B1.val.267]
// CHECK:STDOUT:   assign %b1.var, %.loc19_3
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %b1: ref %B1 = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.b19 = binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.b19 = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %B2 = var %b2.var_patt
// CHECK:STDOUT:   %.loc20_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc20_27.1: %struct_type.base.f5e = struct_literal (%.loc20_26.1)
// CHECK:STDOUT:   %.loc20_27.2: ref %B1 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc20_26.2: ref %ptr.454 = class_element_access %.loc20_27.2, element0
// CHECK:STDOUT:   %.loc20_26.3: init %ptr.454 = initialize_from @B2.%vtable_ptr to %.loc20_26.2 [concrete = constants.%B2.vtable_ptr]
// CHECK:STDOUT:   %.loc20_26.4: init %B1 = class_init (%.loc20_26.3), %.loc20_27.2 [concrete = constants.%B1.val.d2f]
// CHECK:STDOUT:   %.loc20_27.3: init %B1 = converted %.loc20_26.1, %.loc20_26.4 [concrete = constants.%B1.val.d2f]
// CHECK:STDOUT:   %.loc20_27.4: init %B2 = class_init (%.loc20_27.3), %b2.var [concrete = constants.%B2.val.e9e]
// CHECK:STDOUT:   %.loc20_3: init %B2 = converted %.loc20_27.1, %.loc20_27.4 [concrete = constants.%B2.val.e9e]
// CHECK:STDOUT:   assign %b2.var, %.loc20_3
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %b2: ref %B2 = bind_name b2, %b2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc21_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc21_34.1: %struct_type.base.f5e = struct_literal (%.loc21_33.1)
// CHECK:STDOUT:   %.loc21_35.1: %struct_type.base.a0c = struct_literal (%.loc21_34.1)
// CHECK:STDOUT:   %.loc21_35.2: ref %B2 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc21_34.2: ref %B1 = class_element_access %.loc21_35.2, element0
// CHECK:STDOUT:   %.loc21_33.2: ref %ptr.454 = class_element_access %.loc21_34.2, element0
// CHECK:STDOUT:   %.loc21_33.3: init %ptr.454 = initialize_from @C.%vtable_ptr to %.loc21_33.2 [concrete = constants.%C.vtable_ptr]
// CHECK:STDOUT:   %.loc21_33.4: init %B1 = class_init (%.loc21_33.3), %.loc21_34.2 [concrete = constants.%B1.val.b9d]
// CHECK:STDOUT:   %.loc21_34.3: init %B1 = converted %.loc21_33.1, %.loc21_33.4 [concrete = constants.%B1.val.b9d]
// CHECK:STDOUT:   %.loc21_34.4: init %B2 = class_init (%.loc21_34.3), %.loc21_35.2 [concrete = constants.%B2.val.426]
// CHECK:STDOUT:   %.loc21_35.3: init %B2 = converted %.loc21_34.1, %.loc21_34.4 [concrete = constants.%B2.val.426]
// CHECK:STDOUT:   %.loc21_35.4: init %C = class_init (%.loc21_35.3), %c.var [concrete = constants.%C.val]
// CHECK:STDOUT:   %.loc21_3: init %C = converted %.loc21_35.1, %.loc21_35.4 [concrete = constants.%C.val]
// CHECK:STDOUT:   assign %c.var, %.loc21_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   %Op.bound.loc21: <bound method> = bound_method %c.var, constants.%Op.5d7
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.5d7, @Op.2(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %c.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc21: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:   %no_op.loc21: init %empty_tuple.type = call %bound_method.loc21(%addr.loc21)
// CHECK:STDOUT:   %Op.bound.loc20: <bound method> = bound_method %b2.var, constants.%Op.bb0
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.bb0, @Op.2(constants.%B2) [concrete = constants.%Op.specific_fn.a6f]
// CHECK:STDOUT:   %bound_method.loc20: <bound method> = bound_method %b2.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc20: %ptr.afe = addr_of %b2.var
// CHECK:STDOUT:   %no_op.loc20: init %empty_tuple.type = call %bound_method.loc20(%addr.loc20)
// CHECK:STDOUT:   %Op.bound.loc19: <bound method> = bound_method %b1.var, constants.%Op.a5c
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.a5c, @Op.2(constants.%B1) [concrete = constants.%Op.specific_fn.075]
// CHECK:STDOUT:   %bound_method.loc19: <bound method> = bound_method %b1.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc19: %ptr.890 = addr_of %b1.var
// CHECK:STDOUT:   %no_op.loc19: init %empty_tuple.type = call %bound_method.loc19(%addr.loc19)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %C.vtable_ptr: ref %ptr = vtable_ptr @C.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr = vtable_ptr @C.vtable [concrete = constants.%C.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @C.vtable {}
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.loc8 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.loc11 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.047: <bound method> = bound_method %int_3.1ba, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.68c: type = struct_type {.m2: %i32, .m1: %i32} [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.5f2: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.a25: <bound method> = bound_method %int_5.64b, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr, %int_5.0f6, %int_3.822) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.193: type = fn_type @Op.3, @Destroy.impl(%Base) [concrete]
// CHECK:STDOUT:   %Op.345: %Op.type.193 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.11f: type = ptr_type %Base [concrete]
// CHECK:STDOUT:   %Op.specific_fn.083: <specific function> = specific_function %Op.345, @Op.3(%Base) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.3, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.3(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.loc11 [concrete = constants.%F.c41] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %int_32.loc5: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc5: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Base.elem = field_decl m1, element1 [concrete]
// CHECK:STDOUT:   %int_32.loc6: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc6: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %Base.elem = field_decl m2, element2 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.loc8 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [concrete = constants.%struct_type.vptr.m1.m2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .m1 = %.loc5
// CHECK:STDOUT:   .m2 = %.loc6
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc8(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc11() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.7ce = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %i32 = var %i.var_patt
// CHECK:STDOUT:   %int_3.loc12: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc12: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_3.1: <bound method> = bound_method %int_3.loc12, %impl.elem0.loc12 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_3.2: <bound method> = bound_method %int_3.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %bound_method.loc12_3.2(%int_3.loc12) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_3: init %i32 = converted %int_3.loc12, %int.convert_checked.loc12 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %i.var, %.loc12_3
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %i32 = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.bcc = binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.bcc = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %Base = var %b1.var_patt
// CHECK:STDOUT:   %i.ref.loc14_25: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc14_34: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.m2.m1.68c = struct_literal (%i.ref.loc14_25, %i.ref.loc14_34)
// CHECK:STDOUT:   %.loc14_35.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc14_35.3: init %ptr.454 = initialize_from @Base.%vtable_ptr to %.loc14_35.2 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc14_34: %i32 = bind_value %i.ref.loc14_34
// CHECK:STDOUT:   %.loc14_35.4: ref %i32 = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc14_35.5: init %i32 = initialize_from %.loc14_34 to %.loc14_35.4
// CHECK:STDOUT:   %.loc14_25: %i32 = bind_value %i.ref.loc14_25
// CHECK:STDOUT:   %.loc14_35.6: ref %i32 = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc14_35.7: init %i32 = initialize_from %.loc14_25 to %.loc14_35.6
// CHECK:STDOUT:   %.loc14_35.8: init %Base = class_init (%.loc14_35.3, %.loc14_35.5, %.loc14_35.7), %b1.var
// CHECK:STDOUT:   %.loc14_3: init %Base = converted %.loc14_35.1, %.loc14_35.8
// CHECK:STDOUT:   assign %b1.var, %.loc14_3
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.bcc = binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.bcc = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var %b2.var_patt
// CHECK:STDOUT:   %int_3.loc15: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc15_35.1: %struct_type.m2.m1.5f2 = struct_literal (%int_3.loc15, %int_5)
// CHECK:STDOUT:   %.loc15_35.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc15_35.3: init %ptr.454 = initialize_from @Base.%vtable_ptr to %.loc15_35.2 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %impl.elem0.loc15_35.1: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_35.1: <bound method> = bound_method %int_5, %impl.elem0.loc15_35.1 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc15_35.1: <specific function> = specific_function %impl.elem0.loc15_35.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_35.2: <bound method> = bound_method %int_5, %specific_fn.loc15_35.1 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.1: init %i32 = call %bound_method.loc15_35.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.4: init %i32 = converted %int_5, %int.convert_checked.loc15_35.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.5: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc15_35.6: init %i32 = initialize_from %.loc15_35.4 to %.loc15_35.5 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc15_35.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_35.3: <bound method> = bound_method %int_3.loc15, %impl.elem0.loc15_35.2 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc15_35.2: <specific function> = specific_function %impl.elem0.loc15_35.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_35.4: <bound method> = bound_method %int_3.loc15, %specific_fn.loc15_35.2 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.2: init %i32 = call %bound_method.loc15_35.4(%int_3.loc15) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.7: init %i32 = converted %int_3.loc15, %int.convert_checked.loc15_35.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.8: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc15_35.9: init %i32 = initialize_from %.loc15_35.7 to %.loc15_35.8 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.10: init %Base = class_init (%.loc15_35.3, %.loc15_35.6, %.loc15_35.9), %b2.var [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc15_3: init %Base = converted %.loc15_35.1, %.loc15_35.10 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %b2.var, %.loc15_3
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = bind_name b2, %b2.var
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %Base.elem = name_ref m2, @Base.%.loc6 [concrete = @Base.%.loc6]
// CHECK:STDOUT:   %.loc18_5: ref %i32 = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc18: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_9.1: <bound method> = bound_method %int_4, %impl.elem0.loc18 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_9.2: <bound method> = bound_method %int_4, %specific_fn.loc18 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %bound_method.loc18_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc18_9: init %i32 = converted %int_4, %int.convert_checked.loc18 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc18_5, %.loc18_9
// CHECK:STDOUT:   %Op.bound.loc15: <bound method> = bound_method %b2.var, constants.%Op.345
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.345, @Op.3(constants.%Base) [concrete = constants.%Op.specific_fn.083]
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %b2.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc15: %ptr.11f = addr_of %b2.var
// CHECK:STDOUT:   %no_op.loc15: init %empty_tuple.type = call %bound_method.loc15_3(%addr.loc15)
// CHECK:STDOUT:   %Op.bound.loc14: <bound method> = bound_method %b1.var, constants.%Op.345
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.345, @Op.3(constants.%Base) [concrete = constants.%Op.specific_fn.083]
// CHECK:STDOUT:   %bound_method.loc14: <bound method> = bound_method %b1.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc14: %ptr.11f = addr_of %b1.var
// CHECK:STDOUT:   %no_op.loc14: init %empty_tuple.type = call %bound_method.loc14(%addr.loc14)
// CHECK:STDOUT:   %Op.bound.loc12: <bound method> = bound_method %i.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.3: <specific function> = specific_function constants.%Op.e6a, @Op.3(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc12_3.3: <bound method> = bound_method %i.var, %Op.specific_fn.3
// CHECK:STDOUT:   %addr.loc12: %ptr.235 = addr_of %i.var
// CHECK:STDOUT:   %no_op.loc12: init %empty_tuple.type = call %bound_method.loc12_3.3(%addr.loc12)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_impl_without_base_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr.base: type = struct_type {.<vptr>: %ptr.454, .base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.336: <witness> = complete_type_witness %struct_type.vptr.base [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element1 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr.base: type = struct_type {.<vptr>: %ptr.454, .base: %Base} [concrete = constants.%struct_type.vptr.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.base [concrete = constants.%complete_type.336]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {}
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractBase: type = class_type @AbstractBase [concrete]
// CHECK:STDOUT:   %pattern_type.27c: type = pattern_type %AbstractBase [concrete]
// CHECK:STDOUT:   %F.type.85b: type = fn_type @F.loc5 [concrete]
// CHECK:STDOUT:   %F.6e9: %F.type.85b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %AbstractBase.vtable_ptr: ref %ptr.454 = vtable_ptr @AbstractBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %AbstractIntermediate: type = class_type @AbstractIntermediate [concrete]
// CHECK:STDOUT:   %AbstractIntermediate.elem: type = unbound_element_type %AbstractIntermediate, %AbstractBase [concrete]
// CHECK:STDOUT:   %AbstractIntermediate.vtable_ptr: ref %ptr.454 = vtable_ptr @AbstractIntermediate.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.efd: type = struct_type {.base: %AbstractBase} [concrete]
// CHECK:STDOUT:   %complete_type.2d3: <witness> = complete_type_witness %struct_type.base.efd [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %AbstractIntermediate [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc14 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.da5: type = struct_type {.base: %AbstractIntermediate} [concrete]
// CHECK:STDOUT:   %complete_type.f8c: <witness> = complete_type_witness %struct_type.base.da5 [concrete]
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
// CHECK:STDOUT:     .AbstractBase = %AbstractBase.decl
// CHECK:STDOUT:     .AbstractIntermediate = %AbstractIntermediate.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AbstractBase.decl: type = class_decl @AbstractBase [concrete = constants.%AbstractBase] {} {}
// CHECK:STDOUT:   %AbstractIntermediate.decl: type = class_decl @AbstractIntermediate [concrete = constants.%AbstractIntermediate] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractBase {
// CHECK:STDOUT:   %F.decl: %F.type.85b = fn_decl @F.loc5 [concrete = constants.%F.6e9] {
// CHECK:STDOUT:     %self.patt: %pattern_type.27c = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.27c = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AbstractBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractBase [concrete = constants.%AbstractBase]
// CHECK:STDOUT:     %self: %AbstractBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @AbstractBase.vtable [concrete = constants.%AbstractBase.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractBase
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractIntermediate {
// CHECK:STDOUT:   %AbstractBase.ref: type = name_ref AbstractBase, file.%AbstractBase.decl [concrete = constants.%AbstractBase]
// CHECK:STDOUT:   %.loc9: %AbstractIntermediate.elem = base_decl %AbstractBase.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @AbstractIntermediate.vtable [concrete = constants.%AbstractIntermediate.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %AbstractBase} [concrete = constants.%struct_type.base.efd]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.2d3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractIntermediate
// CHECK:STDOUT:   .AbstractBase = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   extend %AbstractBase.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %AbstractIntermediate.ref: type = name_ref AbstractIntermediate, file.%AbstractIntermediate.decl [concrete = constants.%AbstractIntermediate]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %AbstractIntermediate.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc14 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %AbstractIntermediate} [concrete = constants.%struct_type.base.da5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.f8c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .AbstractIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %AbstractIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @AbstractBase.vtable {
// CHECK:STDOUT:   @AbstractBase.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @AbstractIntermediate.vtable {
// CHECK:STDOUT:   constants.%F.6e9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @F.loc5(%self.param: %AbstractBase);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc14(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- virtual_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %VirtualBase: type = class_type @VirtualBase [concrete]
// CHECK:STDOUT:   %pattern_type.ad4: type = pattern_type %VirtualBase [concrete]
// CHECK:STDOUT:   %F.type.e62: type = fn_type @F.loc5 [concrete]
// CHECK:STDOUT:   %F.3e7: %F.type.e62 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %VirtualBase.vtable_ptr: ref %ptr.454 = vtable_ptr @VirtualBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %VirtualIntermediate: type = class_type @VirtualIntermediate [concrete]
// CHECK:STDOUT:   %VirtualIntermediate.elem: type = unbound_element_type %VirtualIntermediate, %VirtualBase [concrete]
// CHECK:STDOUT:   %VirtualIntermediate.vtable_ptr: ref %ptr.454 = vtable_ptr @VirtualIntermediate.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.61e: type = struct_type {.base: %VirtualBase} [concrete]
// CHECK:STDOUT:   %complete_type.f09: <witness> = complete_type_witness %struct_type.base.61e [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %VirtualIntermediate [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc14 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.43c: type = struct_type {.base: %VirtualIntermediate} [concrete]
// CHECK:STDOUT:   %complete_type.fa6: <witness> = complete_type_witness %struct_type.base.43c [concrete]
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
// CHECK:STDOUT:     .VirtualBase = %VirtualBase.decl
// CHECK:STDOUT:     .VirtualIntermediate = %VirtualIntermediate.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %VirtualBase.decl: type = class_decl @VirtualBase [concrete = constants.%VirtualBase] {} {}
// CHECK:STDOUT:   %VirtualIntermediate.decl: type = class_decl @VirtualIntermediate [concrete = constants.%VirtualIntermediate] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualBase {
// CHECK:STDOUT:   %F.decl: %F.type.e62 = fn_decl @F.loc5 [concrete = constants.%F.3e7] {
// CHECK:STDOUT:     %self.patt: %pattern_type.ad4 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.ad4 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %VirtualBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%VirtualBase [concrete = constants.%VirtualBase]
// CHECK:STDOUT:     %self: %VirtualBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @VirtualBase.vtable [concrete = constants.%VirtualBase.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%VirtualBase
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualIntermediate {
// CHECK:STDOUT:   %VirtualBase.ref: type = name_ref VirtualBase, file.%VirtualBase.decl [concrete = constants.%VirtualBase]
// CHECK:STDOUT:   %.loc9: %VirtualIntermediate.elem = base_decl %VirtualBase.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @VirtualIntermediate.vtable [concrete = constants.%VirtualIntermediate.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %VirtualBase} [concrete = constants.%struct_type.base.61e]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.f09]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%VirtualIntermediate
// CHECK:STDOUT:   .VirtualBase = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   extend %VirtualBase.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %VirtualIntermediate.ref: type = name_ref VirtualIntermediate, file.%VirtualIntermediate.decl [concrete = constants.%VirtualIntermediate]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %VirtualIntermediate.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc14 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %VirtualIntermediate} [concrete = constants.%struct_type.base.43c]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.fa6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .VirtualIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %VirtualIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualBase.vtable {
// CHECK:STDOUT:   @VirtualBase.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualIntermediate.vtable {
// CHECK:STDOUT:   constants.%F.3e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc5(%self.param: %VirtualBase);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc14(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.loc5 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc17 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.loc5 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc9: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc17 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.7ce = value_param_pattern %v.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %.loc17: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: %i32 = bind_name v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc5(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc17(%self.param: %Derived, %v.param: %i32);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impl_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e40: type = facet_type <@ImplicitAs, @ImplicitAs(%T1)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness file.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.682: type = pattern_type %T2 [concrete]
// CHECK:STDOUT:   %pattern_type.28b: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %Convert.type.c41: type = fn_type @Convert.loc11 [concrete]
// CHECK:STDOUT:   %Convert.f35: %Convert.type.c41 = struct_value () [concrete]
// CHECK:STDOUT:   %T1.val: %T1 = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.loc17 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc29 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.15c: <witness> = complete_type_witness %struct_type.base [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .T2 = %T2.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %T2.decl: type = class_decl @T2 [concrete = constants.%T2] {} {}
// CHECK:STDOUT:   impl_decl @ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T1)> [concrete = constants.%ImplicitAs.type.e40]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (@ImplicitAs.impl.%Convert.decl), @ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @ImplicitAs.impl: %T2.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c41 = fn_decl @Convert.loc11 [concrete = constants.%Convert.f35] {
// CHECK:STDOUT:     %self.patt: %pattern_type.682 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.682 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.28b = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.28b = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %self.param: %T2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @ImplicitAs.impl.%T2.ref [concrete = constants.%T2]
// CHECK:STDOUT:     %self: %T2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.loc17 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %pattern_type.bcc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.bcc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.28b = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.28b = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc21: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc29 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.682 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.682 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Convert.loc11(%self.param: %T2) -> %return.param: %T1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_13.2: init %T1 = class_init (), %return [concrete = constants.%T1.val]
// CHECK:STDOUT:   %.loc12_14: init %T1 = converted %.loc12_13.1, %.loc12_13.2 [concrete = constants.%T1.val]
// CHECK:STDOUT:   return %.loc12_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.loc17(%self.param: %Base) -> %return.param: %T1;
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc29(%self.param: %Derived) -> %return.param: %T2;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T.loc7_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Base(%T.loc7_17.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.specific_fn.loc13_1.2: <specific function> = specific_function %F, @F(%T.loc7_17.2) [symbolic = %F.specific_fn.loc13_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %vtable_ptr.loc13_1.2: ref %ptr = vtable_ptr @Base.vtable, @Base(%T.loc7_17.2) [symbolic = %vtable_ptr.loc13_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type) = fn_decl @F [symbolic = @Base.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc12_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc12_22.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_22.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%Base (%Base) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc13_1.1: <specific function> = specific_function %F.decl, @F(constants.%T) [symbolic = %F.specific_fn.loc13_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %vtable_ptr.loc13_1.1: ref %ptr = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc13_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc13_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(@Base.%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.9f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.%Base (%Base));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc8, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc8(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.573: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F.loc8, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3bf: type = pattern_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.28b: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %F.specific_fn.210: <specific function> = specific_function %F.d25, @F.loc8(%T1) [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.bfe: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc13 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.fda: type = struct_type {.base: %Base.ea5} [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %struct_type.base.fda [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T.loc7_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc8, @Base(%T.loc7_17.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2: <specific function> = specific_function %F, @F.loc8(%T.loc7_17.2) [symbolic = %F.specific_fn.loc9_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc7_17.2) [symbolic = %vtable_ptr.loc9_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc8 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc8.%pattern_type.loc8_16 (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc8.%pattern_type.loc8_16 (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc8.%pattern_type.loc8_28 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc8.%pattern_type.loc8_28 (%pattern_type.7dc) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc8.%Base (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc8_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc8_22.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_22.2 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc8.%Base (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc8.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Base.%T.loc7_17.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.loc8.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc9_1.1: <specific function> = specific_function %F.decl, @F.loc8(constants.%T) [symbolic = %F.specific_fn.loc9_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc9_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc9_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:   %.loc12: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc13 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.28b = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.28b = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %t.param: %T1 = value_param call_param1
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %t: %T1 = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.ea5} [concrete = constants.%struct_type.base.fda]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.65a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc9_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F.loc8(@Base.%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type.loc8_16: type = pattern_type %Base [symbolic = %pattern_type.loc8_16 (constants.%pattern_type.9f7)]
// CHECK:STDOUT:   %pattern_type.loc8_28: type = pattern_type %T [symbolic = %pattern_type.loc8_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_20: <witness> = require_complete_type %Base [symbolic = %require_complete.loc8_20 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %require_complete.loc8_29: <witness> = require_complete_type %T [symbolic = %require_complete.loc8_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.loc8.%Base (%Base.370), %t.param: @F.loc8.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc13(%self.param: %Derived, %t.param: %T1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2 => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2 => constants.%Base.vtable_ptr.573
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc8(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %pattern_type.loc8_16 => constants.%pattern_type.9f7
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_20 => constants.%require_complete.97d
// CHECK:STDOUT:   %require_complete.loc8_29 => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2 => constants.%F.specific_fn.210
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2 => constants.%Base.vtable_ptr.bfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc8(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.ea5
// CHECK:STDOUT:   %pattern_type.loc8_16 => constants.%pattern_type.3bf
// CHECK:STDOUT:   %pattern_type.loc8_28 => constants.%pattern_type.28b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_20 => constants.%complete_type.513
// CHECK:STDOUT:   %require_complete.loc8_29 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %F.type.ba7: type = fn_type @F.loc9 [concrete]
// CHECK:STDOUT:   %F.1a5: %F.type.ba7 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %F.type.834: type = fn_type @F.loc23 [concrete]
// CHECK:STDOUT:   %F.a48: %F.type.834 = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete]
// CHECK:STDOUT:   %complete_type.e14: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .T2 = %T2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %T2.decl: type = class_decl @T2 [concrete = constants.%T2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %F.decl: %F.type.ba7 = fn_decl @F.loc9 [concrete = constants.%F.1a5] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %.loc18: %T2.elem = base_decl %T1.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.834 = fn_decl @F.loc23 [concrete = constants.%F.a48] {} {}
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.e14]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %T1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc9();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc23();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_addr_self_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %ptr.87b: type = ptr_type %T1 [concrete]
// CHECK:STDOUT:   %pattern_type.a36: type = pattern_type %ptr.87b [concrete]
// CHECK:STDOUT:   %F1.type.b96: type = fn_type @F1.loc5 [concrete]
// CHECK:STDOUT:   %F1.765: %F1.type.b96 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.vtable_ptr: ref %ptr.454 = vtable_ptr @T1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %ptr.63e: type = ptr_type %T2 [concrete]
// CHECK:STDOUT:   %pattern_type.fb8: type = pattern_type %ptr.63e [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %F1.type.b0d: type = fn_type @F1.loc17 [concrete]
// CHECK:STDOUT:   %F1.0ce: %F1.type.b0d = struct_value () [concrete]
// CHECK:STDOUT:   %T2.vtable_ptr: ref %ptr.454 = vtable_ptr @T2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete]
// CHECK:STDOUT:   %complete_type.e14: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .T2 = %T2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %T2.decl: type = class_decl @T2 [concrete = constants.%T2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %F1.decl: %F1.type.b96 = fn_decl @F1.loc5 [concrete = constants.%F1.765] {
// CHECK:STDOUT:     %self.patt: %pattern_type.a36 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a36 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.87b = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %ptr [concrete = constants.%ptr.87b] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%T1 [concrete = constants.%T1]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.87b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.87b = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @T1.vtable [concrete = constants.%T1.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %.loc9: %T2.elem = base_decl %T1.ref, element0 [concrete]
// CHECK:STDOUT:   %F1.decl: %F1.type.b0d = fn_decl @F1.loc17 [concrete = constants.%F1.0ce] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc17_14: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.63e = value_param call_param0
// CHECK:STDOUT:     %.loc17_29: type = splice_block %ptr [concrete = constants.%ptr.63e] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%T2 [concrete = constants.%T2]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.63e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.63e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @T2.vtable [concrete = constants.%T2.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete = constants.%struct_type.base]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.e14]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT:   extend %T1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T2.vtable {
// CHECK:STDOUT:   @T2.%F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F1.loc5(%self.param: %ptr.87b);
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F1.loc17(%self.param: %ptr.63e);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %pattern_type.28b: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %self.patt: %pattern_type.28b = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.28b = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %T1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%T1 [concrete = constants.%T1]
// CHECK:STDOUT:     %self: %T1 = bind_name self, %self.param
// CHECK:STDOUT:     %T.loc9_28.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_28.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(%T.loc9_28.2: type) {
// CHECK:STDOUT:   %T.loc9_28.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_28.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: %T1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_28.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual_in_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T.8b3) [symbolic]
// CHECK:STDOUT:   %pattern_type.48e: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %T.336: type = bind_symbolic_name T, 1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T.8b3) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: %T1.type = class_decl @T1 [concrete = constants.%T1.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T.loc4_15.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @T1.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @T1.%F.type (%F.type) = fn_decl @F [symbolic = @T1.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type (%pattern_type.48e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type (%pattern_type.48e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc9_22.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc9_22.2: type = specific_constant constants.%T1, @T1(constants.%T.8b3) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_22.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%T1 (%T1) = bind_name self, %self.param
// CHECK:STDOUT:       %T.loc9_28.2: type = bind_symbolic_name T, 1 [symbolic = %T.loc9_28.1 (constants.%T.336)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(@T1.%T.loc4_15.1: type, %T.loc9_28.2: type) {
// CHECK:STDOUT:   %T.loc9_22: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_22 (constants.%T.8b3)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T.loc9_22) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.48e)]
// CHECK:STDOUT:   %T.loc9_28.1: type = bind_symbolic_name T, 1 [symbolic = %T.loc9_28.1 (constants.%T.336)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.%T1 (%T1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3, constants.%T.336) {
// CHECK:STDOUT:   %T.loc9_22 => constants.%T.8b3
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48e
// CHECK:STDOUT:   %T.loc9_28.1 => constants.%T.336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_with_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.48e: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %T1.vtable_ptr: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: %T1.type = class_decl @T1 [concrete = constants.%T1.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T.loc4_15.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @T1.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F(%T.loc4_15.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(%T.loc4_15.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%T1.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @T1.%F.type (%F.type) = fn_decl @F [symbolic = @T1.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type (%pattern_type.48e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type (%pattern_type.48e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%T1 (%T1) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%T1.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(@T1.%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.48e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.%T1 (%T1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%T1.vtable_ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_dependent_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.48e: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %T1.vtable_ptr: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.86d: <witness> = require_complete_type %T1 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: %T1.type = class_decl @T1 [concrete = constants.%T1.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @T1(%T.loc4_15.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @T1.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F(%T.loc4_15.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(%T.loc4_15.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%T1.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @T1.%F.type (%F.type) = fn_decl @F [symbolic = @T1.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type.loc5_16 (%pattern_type.48e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type.loc5_16 (%pattern_type.48e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.%pattern_type.loc5_28 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.%pattern_type.loc5_28 (%pattern_type.7dc) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%T1 (%T1) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @T1.%T.loc4_15.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @T1.vtable, @T1(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%T1.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(@T1.%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %T1 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.48e)]
// CHECK:STDOUT:   %pattern_type.loc5_28: type = pattern_type %T [symbolic = %pattern_type.loc5_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20: <witness> = require_complete_type %T1 [symbolic = %require_complete.loc5_20 (constants.%require_complete.86d)]
// CHECK:STDOUT:   %require_complete.loc5_29: <witness> = require_complete_type %T [symbolic = %require_complete.loc5_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.%T1 (%T1), %t.param: @F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%T1.vtable_ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.48e
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%require_complete.86d
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- vtable_import_unneeded.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Modifiers.import_ref.39e92f.2 = import_ref Modifiers//default, loc6_1, unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Modifiers.import_ref.1f3 = import_ref Modifiers//default, inst18 [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.2cc = import_ref Modifiers//default, loc5_29, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %b.patt: %pattern_type = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.param_patt: %pattern_type = value_param_pattern %b.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Base = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:       %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:       %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %Base = bind_name b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_ptr = imports.%Modifiers.import_ref.39e92f.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_derived_from_nongeneric.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonGenericBase: type = class_type @NonGenericBase [concrete]
// CHECK:STDOUT:   %pattern_type.0b8: type = pattern_type %NonGenericBase [concrete]
// CHECK:STDOUT:   %F1.type: type = fn_type @F1 [concrete]
// CHECK:STDOUT:   %F1: %F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %F2.type.809: type = fn_type @F2.loc6 [concrete]
// CHECK:STDOUT:   %F2.053: %F2.type.809 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %NonGenericBase.vtable_ptr: ref %ptr.454 = vtable_ptr @NonGenericBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericDerived.type: type = generic_class_type @GenericDerived [concrete]
// CHECK:STDOUT:   %GenericDerived.generic: %GenericDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.elem: type = unbound_element_type %GenericDerived, %NonGenericBase [symbolic]
// CHECK:STDOUT:   %pattern_type.061: type = pattern_type %GenericDerived [symbolic]
// CHECK:STDOUT:   %F2.type.e6e: type = fn_type @F2.loc11, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %F2.5da: %F2.type.e6e = struct_value () [symbolic]
// CHECK:STDOUT:   %F3.type: type = fn_type @F3, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %F3: %F3.type = struct_value () [symbolic]
// CHECK:STDOUT:   %F2.specific_fn: <specific function> = specific_function %F2.5da, @F2.loc11(%T) [symbolic]
// CHECK:STDOUT:   %F3.specific_fn: <specific function> = specific_function %F3, @F3(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.vtable_ptr: ref %ptr.454 = vtable_ptr @GenericDerived.vtable, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.base.432: type = struct_type {.base: %NonGenericBase} [concrete]
// CHECK:STDOUT:   %complete_type.099: <witness> = complete_type_witness %struct_type.base.432 [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericDerived [symbolic]
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
// CHECK:STDOUT:     .NonGenericBase = %NonGenericBase.decl
// CHECK:STDOUT:     .GenericDerived = %GenericDerived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NonGenericBase.decl: type = class_decl @NonGenericBase [concrete = constants.%NonGenericBase] {} {}
// CHECK:STDOUT:   %GenericDerived.decl: %GenericDerived.type = class_decl @GenericDerived [concrete = constants.%GenericDerived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc9_27.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_27.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonGenericBase {
// CHECK:STDOUT:   %F1.decl: %F1.type = fn_decl @F1 [concrete = constants.%F1] {
// CHECK:STDOUT:     %self.patt: %pattern_type.0b8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0b8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericBase [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %self: %NonGenericBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F2.decl: %F2.type.809 = fn_decl @F2.loc6 [concrete = constants.%F2.053] {
// CHECK:STDOUT:     %self.patt: %pattern_type.0b8 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0b8 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericBase [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %self: %NonGenericBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @NonGenericBase.vtable [concrete = constants.%NonGenericBase.vtable_ptr]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonGenericBase
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT:   .F2 = %F2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericDerived(%T.loc9_27.1: type) {
// CHECK:STDOUT:   %T.loc9_27.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_27.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T.loc9_27.2) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %GenericDerived.elem: type = unbound_element_type %GenericDerived, constants.%NonGenericBase [symbolic = %GenericDerived.elem (constants.%GenericDerived.elem)]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2.loc11, @GenericDerived(%T.loc9_27.2) [symbolic = %F2.type (constants.%F2.type.e6e)]
// CHECK:STDOUT:   %F2: @GenericDerived.%F2.type (%F2.type.e6e) = struct_value () [symbolic = %F2 (constants.%F2.5da)]
// CHECK:STDOUT:   %F3.type: type = fn_type @F3, @GenericDerived(%T.loc9_27.2) [symbolic = %F3.type (constants.%F3.type)]
// CHECK:STDOUT:   %F3: @GenericDerived.%F3.type (%F3.type) = struct_value () [symbolic = %F3 (constants.%F3)]
// CHECK:STDOUT:   %F2.specific_fn.loc13_1.2: <specific function> = specific_function %F2, @F2.loc11(%T.loc9_27.2) [symbolic = %F2.specific_fn.loc13_1.2 (constants.%F2.specific_fn)]
// CHECK:STDOUT:   %F3.specific_fn.loc13_1.2: <specific function> = specific_function %F3, @F3(%T.loc9_27.2) [symbolic = %F3.specific_fn.loc13_1.2 (constants.%F3.specific_fn)]
// CHECK:STDOUT:   %vtable_ptr.loc13_1.2: ref %ptr.454 = vtable_ptr @GenericDerived.vtable, @GenericDerived(%T.loc9_27.2) [symbolic = %vtable_ptr.loc13_1.2 (constants.%GenericDerived.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %NonGenericBase.ref: type = name_ref NonGenericBase, file.%NonGenericBase.decl [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %.loc10: @GenericDerived.%GenericDerived.elem (%GenericDerived.elem) = base_decl %NonGenericBase.ref, element0 [concrete]
// CHECK:STDOUT:     %F2.decl: @GenericDerived.%F2.type (%F2.type.e6e) = fn_decl @F2.loc11 [symbolic = @GenericDerived.%F2 (constants.%F2.5da)] {
// CHECK:STDOUT:       %self.patt: @F2.loc11.%pattern_type (%pattern_type.061) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F2.loc11.%pattern_type (%pattern_type.061) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F2.loc11.%GenericDerived (%GenericDerived) = value_param call_param0
// CHECK:STDOUT:       %.loc11_20.1: type = splice_block %Self.ref [symbolic = %GenericDerived (constants.%GenericDerived)] {
// CHECK:STDOUT:         %.loc11_20.2: type = specific_constant constants.%GenericDerived, @GenericDerived(constants.%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_20.2 [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F2.loc11.%GenericDerived (%GenericDerived) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F3.decl: @GenericDerived.%F3.type (%F3.type) = fn_decl @F3 [symbolic = @GenericDerived.%F3 (constants.%F3)] {
// CHECK:STDOUT:       %self.patt: @F3.%pattern_type (%pattern_type.061) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F3.%pattern_type (%pattern_type.061) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F3.%GenericDerived (%GenericDerived) = value_param call_param0
// CHECK:STDOUT:       %.loc12_23.1: type = splice_block %Self.ref [symbolic = %GenericDerived (constants.%GenericDerived)] {
// CHECK:STDOUT:         %.loc12_23.2: type = specific_constant constants.%GenericDerived, @GenericDerived(constants.%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_23.2 [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F3.%GenericDerived (%GenericDerived) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F2.specific_fn.loc13_1.1: <specific function> = specific_function %F2.decl, @F2.loc11(constants.%T) [symbolic = %F2.specific_fn.loc13_1.2 (constants.%F2.specific_fn)]
// CHECK:STDOUT:     %F3.specific_fn.loc13_1.1: <specific function> = specific_function %F3.decl, @F3(constants.%T) [symbolic = %F3.specific_fn.loc13_1.2 (constants.%F3.specific_fn)]
// CHECK:STDOUT:     %vtable_ptr.loc13_1.1: ref %ptr.454 = vtable_ptr @GenericDerived.vtable, @GenericDerived(constants.%T) [symbolic = %vtable_ptr.loc13_1.2 (constants.%GenericDerived.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.base: type = struct_type {.base: %NonGenericBase} [concrete = constants.%struct_type.base.432]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.099]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc13_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericDerived
// CHECK:STDOUT:     .NonGenericBase = <poisoned>
// CHECK:STDOUT:     .base = %.loc10
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:     .F3 = %F3.decl
// CHECK:STDOUT:     extend %NonGenericBase.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonGenericBase.vtable {
// CHECK:STDOUT:   @NonGenericBase.%F1.decl
// CHECK:STDOUT:   @NonGenericBase.%F2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @GenericDerived.vtable {
// CHECK:STDOUT:   constants.%F1
// CHECK:STDOUT:   @GenericDerived.%F2.specific_fn.loc13_1.1
// CHECK:STDOUT:   @GenericDerived.%F3.specific_fn.loc13_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F1(%self.param: %NonGenericBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F2.loc6(%self.param: %NonGenericBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl fn @F2.loc11(@GenericDerived.%T.loc9_27.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericDerived [symbolic = %pattern_type (constants.%pattern_type.061)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericDerived [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl fn(%self.param: @F2.loc11.%GenericDerived (%GenericDerived)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F3(@GenericDerived.%T.loc9_27.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericDerived [symbolic = %pattern_type (constants.%pattern_type.061)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericDerived [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F3.%GenericDerived (%GenericDerived)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc9_27.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %GenericDerived.elem => constants.%GenericDerived.elem
// CHECK:STDOUT:   %F2.type => constants.%F2.type.e6e
// CHECK:STDOUT:   %F2 => constants.%F2.5da
// CHECK:STDOUT:   %F3.type => constants.%F3.type
// CHECK:STDOUT:   %F3 => constants.%F3
// CHECK:STDOUT:   %F2.specific_fn.loc13_1.2 => constants.%F2.specific_fn
// CHECK:STDOUT:   %F3.specific_fn.loc13_1.2 => constants.%F3.specific_fn
// CHECK:STDOUT:   %vtable_ptr.loc13_1.2 => constants.%GenericDerived.vtable_ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.loc11(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.061
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F3(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.061
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nongeneric_derived_from_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %GenericBase.type: type = generic_class_type @GenericBase [concrete]
// CHECK:STDOUT:   %GenericBase.generic: %GenericBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericBase.018: type = class_type @GenericBase, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.34e: type = pattern_type %GenericBase.018 [symbolic]
// CHECK:STDOUT:   %F1.type.2af: type = fn_type @F1, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %F1.80a: %F1.type.2af = struct_value () [symbolic]
// CHECK:STDOUT:   %F2.type.2ca: type = fn_type @F2.loc6, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %F2.414: %F2.type.2ca = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F1.specific_fn.aea: <specific function> = specific_function %F1.80a, @F1(%T) [symbolic]
// CHECK:STDOUT:   %F2.specific_fn.758: <specific function> = specific_function %F2.414, @F2.loc6(%T) [symbolic]
// CHECK:STDOUT:   %GenericBase.vtable_ptr.e41: ref %ptr.454 = vtable_ptr @GenericBase.vtable, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase.018 [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %NonGenericDerived: type = class_type @NonGenericDerived [concrete]
// CHECK:STDOUT:   %GenericBase.f84: type = class_type @GenericBase, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %F1.type.648: type = fn_type @F1, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %F1.4d3: %F1.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %F2.type.d79: type = fn_type @F2.loc6, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %F2.d59: %F2.type.d79 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7a9: type = pattern_type %GenericBase.f84 [concrete]
// CHECK:STDOUT:   %F1.specific_fn.094: <specific function> = specific_function %F1.4d3, @F1(%T1) [concrete]
// CHECK:STDOUT:   %F2.specific_fn.b8b: <specific function> = specific_function %F2.d59, @F2.loc6(%T1) [concrete]
// CHECK:STDOUT:   %GenericBase.vtable_ptr.d41: ref %ptr.454 = vtable_ptr @GenericBase.vtable, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %NonGenericDerived.elem: type = unbound_element_type %NonGenericDerived, %GenericBase.f84 [concrete]
// CHECK:STDOUT:   %pattern_type.5fc: type = pattern_type %NonGenericDerived [concrete]
// CHECK:STDOUT:   %F2.type.c10: type = fn_type @F2.loc13 [concrete]
// CHECK:STDOUT:   %F2.deb: %F2.type.c10 = struct_value () [concrete]
// CHECK:STDOUT:   %F3.type: type = fn_type @F3 [concrete]
// CHECK:STDOUT:   %F3: %F3.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonGenericDerived.vtable_ptr: ref %ptr.454 = vtable_ptr @NonGenericDerived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.29a: type = struct_type {.base: %GenericBase.f84} [concrete]
// CHECK:STDOUT:   %complete_type.b6e: <witness> = complete_type_witness %struct_type.base.29a [concrete]
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
// CHECK:STDOUT:     .GenericBase = %GenericBase.decl
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .NonGenericDerived = %NonGenericDerived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %GenericBase.decl: %GenericBase.type = class_decl @GenericBase [concrete = constants.%GenericBase.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_24.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_24.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %NonGenericDerived.decl: type = class_decl @NonGenericDerived [concrete = constants.%NonGenericDerived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericBase(%T.loc4_24.1: type) {
// CHECK:STDOUT:   %T.loc4_24.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_24.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F1.type: type = fn_type @F1, @GenericBase(%T.loc4_24.2) [symbolic = %F1.type (constants.%F1.type.2af)]
// CHECK:STDOUT:   %F1: @GenericBase.%F1.type (%F1.type.2af) = struct_value () [symbolic = %F1 (constants.%F1.80a)]
// CHECK:STDOUT:   %F2.type: type = fn_type @F2.loc6, @GenericBase(%T.loc4_24.2) [symbolic = %F2.type (constants.%F2.type.2ca)]
// CHECK:STDOUT:   %F2: @GenericBase.%F2.type (%F2.type.2ca) = struct_value () [symbolic = %F2 (constants.%F2.414)]
// CHECK:STDOUT:   %F1.specific_fn.loc7_1.2: <specific function> = specific_function %F1, @F1(%T.loc4_24.2) [symbolic = %F1.specific_fn.loc7_1.2 (constants.%F1.specific_fn.aea)]
// CHECK:STDOUT:   %F2.specific_fn.loc7_1.2: <specific function> = specific_function %F2, @F2.loc6(%T.loc4_24.2) [symbolic = %F2.specific_fn.loc7_1.2 (constants.%F2.specific_fn.758)]
// CHECK:STDOUT:   %vtable_ptr.loc7_1.2: ref %ptr.454 = vtable_ptr @GenericBase.vtable, @GenericBase(%T.loc4_24.2) [symbolic = %vtable_ptr.loc7_1.2 (constants.%GenericBase.vtable_ptr.e41)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F1.decl: @GenericBase.%F1.type (%F1.type.2af) = fn_decl @F1 [symbolic = @GenericBase.%F1 (constants.%F1.80a)] {
// CHECK:STDOUT:       %self.patt: @F1.%pattern_type (%pattern_type.34e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F1.%pattern_type (%pattern_type.34e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F1.%GenericBase (%GenericBase.018) = value_param call_param0
// CHECK:STDOUT:       %.loc5_23.1: type = splice_block %Self.ref [symbolic = %GenericBase (constants.%GenericBase.018)] {
// CHECK:STDOUT:         %.loc5_23.2: type = specific_constant constants.%GenericBase.018, @GenericBase(constants.%T) [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_23.2 [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F1.%GenericBase (%GenericBase.018) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F2.decl: @GenericBase.%F2.type (%F2.type.2ca) = fn_decl @F2.loc6 [symbolic = @GenericBase.%F2 (constants.%F2.414)] {
// CHECK:STDOUT:       %self.patt: @F2.loc6.%pattern_type (%pattern_type.34e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F2.loc6.%pattern_type (%pattern_type.34e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F2.loc6.%GenericBase (%GenericBase.018) = value_param call_param0
// CHECK:STDOUT:       %.loc6_23.1: type = splice_block %Self.ref [symbolic = %GenericBase (constants.%GenericBase.018)] {
// CHECK:STDOUT:         %.loc6_23.2: type = specific_constant constants.%GenericBase.018, @GenericBase(constants.%T) [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_23.2 [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F2.loc6.%GenericBase (%GenericBase.018) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F1.specific_fn.loc7_1.1: <specific function> = specific_function %F1.decl, @F1(constants.%T) [symbolic = %F1.specific_fn.loc7_1.2 (constants.%F1.specific_fn.aea)]
// CHECK:STDOUT:     %F2.specific_fn.loc7_1.1: <specific function> = specific_function %F2.decl, @F2.loc6(constants.%T) [symbolic = %F2.specific_fn.loc7_1.2 (constants.%F2.specific_fn.758)]
// CHECK:STDOUT:     %vtable_ptr.loc7_1.1: ref %ptr.454 = vtable_ptr @GenericBase.vtable, @GenericBase(constants.%T) [symbolic = %vtable_ptr.loc7_1.2 (constants.%GenericBase.vtable_ptr.e41)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc7_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericBase.018
// CHECK:STDOUT:     .F1 = %F1.decl
// CHECK:STDOUT:     .F2 = %F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonGenericDerived {
// CHECK:STDOUT:   %GenericBase.ref: %GenericBase.type = name_ref GenericBase, file.%GenericBase.decl [concrete = constants.%GenericBase.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(constants.%T1) [concrete = constants.%GenericBase.f84]
// CHECK:STDOUT:   %.loc12: %NonGenericDerived.elem = base_decl %GenericBase, element0 [concrete]
// CHECK:STDOUT:   %F2.decl: %F2.type.c10 = fn_decl @F2.loc13 [concrete = constants.%F2.deb] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5fc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5fc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericDerived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericDerived [concrete = constants.%NonGenericDerived]
// CHECK:STDOUT:     %self: %NonGenericDerived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F3.decl: %F3.type = fn_decl @F3 [concrete = constants.%F3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.5fc = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.5fc = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericDerived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericDerived [concrete = constants.%NonGenericDerived]
// CHECK:STDOUT:     %self: %NonGenericDerived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @NonGenericDerived.vtable [concrete = constants.%NonGenericDerived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %GenericBase.f84} [concrete = constants.%struct_type.base.29a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.b6e]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonGenericDerived
// CHECK:STDOUT:   .GenericBase = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .F2 = %F2.decl
// CHECK:STDOUT:   .F3 = %F3.decl
// CHECK:STDOUT:   extend %GenericBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @GenericBase.vtable {
// CHECK:STDOUT:   @GenericBase.%F1.specific_fn.loc7_1.1
// CHECK:STDOUT:   @GenericBase.%F2.specific_fn.loc7_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonGenericDerived.vtable {
// CHECK:STDOUT:   constants.%F1.specific_fn.094
// CHECK:STDOUT:   @NonGenericDerived.%F2.decl
// CHECK:STDOUT:   @NonGenericDerived.%F3.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F1(@GenericBase.%T.loc4_24.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(%T) [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericBase [symbolic = %pattern_type (constants.%pattern_type.34e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F1.%GenericBase (%GenericBase.018)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F2.loc6(@GenericBase.%T.loc4_24.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(%T) [symbolic = %GenericBase (constants.%GenericBase.018)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericBase [symbolic = %pattern_type (constants.%pattern_type.34e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F2.loc6.%GenericBase (%GenericBase.018)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F2.loc13(%self.param: %NonGenericDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F3(%self.param: %NonGenericDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase(constants.%T) {
// CHECK:STDOUT:   %T.loc4_24.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F1.type => constants.%F1.type.2af
// CHECK:STDOUT:   %F1 => constants.%F1.80a
// CHECK:STDOUT:   %F2.type => constants.%F2.type.2ca
// CHECK:STDOUT:   %F2 => constants.%F2.414
// CHECK:STDOUT:   %F1.specific_fn.loc7_1.2 => constants.%F1.specific_fn.aea
// CHECK:STDOUT:   %F2.specific_fn.loc7_1.2 => constants.%F2.specific_fn.758
// CHECK:STDOUT:   %vtable_ptr.loc7_1.2 => constants.%GenericBase.vtable_ptr.e41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.018
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.34e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.loc6(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.018
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.34e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase(constants.%T1) {
// CHECK:STDOUT:   %T.loc4_24.2 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F1.type => constants.%F1.type.648
// CHECK:STDOUT:   %F1 => constants.%F1.4d3
// CHECK:STDOUT:   %F2.type => constants.%F2.type.d79
// CHECK:STDOUT:   %F2 => constants.%F2.d59
// CHECK:STDOUT:   %F1.specific_fn.loc7_1.2 => constants.%F1.specific_fn.094
// CHECK:STDOUT:   %F2.specific_fn.loc7_1.2 => constants.%F2.specific_fn.b8b
// CHECK:STDOUT:   %vtable_ptr.loc7_1.2 => constants.%GenericBase.vtable_ptr.d41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F1(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.f84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F2.loc6(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.f84
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_specifically.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %ptr.b7c: type = ptr_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.8d4: type = pattern_type %ptr.b7c [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc5, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc5(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.573: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %require_complete.cbc: <witness> = require_complete_type %ptr.b7c [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %D1: type = class_type @D1 [concrete]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F.loc5, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3bf: type = pattern_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %ptr.184: type = ptr_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.04d: type = pattern_type %ptr.184 [concrete]
// CHECK:STDOUT:   %F.specific_fn.210: <specific function> = specific_function %F.d25, @F.loc5(%T1) [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.bfe: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T1) [concrete]
// CHECK:STDOUT:   %D1.elem: type = unbound_element_type %D1, %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.550: type = pattern_type %D1 [concrete]
// CHECK:STDOUT:   %F.type.af3: type = fn_type @F.loc10 [concrete]
// CHECK:STDOUT:   %F.87b: %F.type.af3 = struct_value () [concrete]
// CHECK:STDOUT:   %D1.vtable_ptr: ref %ptr.454 = vtable_ptr @D1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.fda: type = struct_type {.base: %Base.ea5} [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %struct_type.base.fda [concrete]
// CHECK:STDOUT:   %complete_type.093: <witness> = complete_type_witness %ptr.184 [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .D1 = %D1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %D1.decl: type = class_decl @D1 [concrete = constants.%D1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc5, @Base(%T.loc4_17.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F.loc5(%T.loc4_17.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc4_17.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc5 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.8d4) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.8d4) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc5.%Base.loc5_22 (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %Base.loc5_22 (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base.loc5_22 (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %Base.loc5_22 (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc5.%Base.loc5_22 (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc5.%ptr.loc5_38.1 (%ptr.b7c) = value_param call_param1
// CHECK:STDOUT:       %.loc5_38: type = splice_block %ptr.loc5_38.2 [symbolic = %ptr.loc5_38.1 (constants.%ptr.b7c)] {
// CHECK:STDOUT:         %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Base.%T.loc4_17.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %Base.loc5_37: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc5_22 (constants.%Base.370)]
// CHECK:STDOUT:         %ptr.loc5_38.2: type = ptr_type %Base.loc5_37 [symbolic = %ptr.loc5_38.1 (constants.%ptr.b7c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @F.loc5.%ptr.loc5_38.1 (%ptr.b7c) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F.loc5(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:   %.loc9: %D1.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.af3 = fn_decl @F.loc10 [concrete = constants.%F.87b] {
// CHECK:STDOUT:     %self.patt: %pattern_type.550 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.550 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.04d = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.04d = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D1 [concrete = constants.%D1]
// CHECK:STDOUT:     %self: %D1 = bind_name self, %self.param
// CHECK:STDOUT:     %t.param: %ptr.184 = value_param call_param1
// CHECK:STDOUT:     %.loc10: type = splice_block %ptr [concrete = constants.%ptr.184] {
// CHECK:STDOUT:       %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:       %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:       %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:       %ptr: type = ptr_type %Base [concrete = constants.%ptr.184]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %ptr.184 = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @D1.vtable [concrete = constants.%D1.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.ea5} [concrete = constants.%struct_type.base.fda]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.65a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @D1.vtable {
// CHECK:STDOUT:   @D1.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F.loc5(@Base.%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base.loc5_22: type = class_type @Base, @Base(%T) [symbolic = %Base.loc5_22 (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Base.loc5_22 [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.9f7)]
// CHECK:STDOUT:   %ptr.loc5_38.1: type = ptr_type %Base.loc5_22 [symbolic = %ptr.loc5_38.1 (constants.%ptr.b7c)]
// CHECK:STDOUT:   %pattern_type.loc5_28: type = pattern_type %ptr.loc5_38.1 [symbolic = %pattern_type.loc5_28 (constants.%pattern_type.8d4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20: <witness> = require_complete_type %Base.loc5_22 [symbolic = %require_complete.loc5_20 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %require_complete.loc5_29: <witness> = require_complete_type %ptr.loc5_38.1 [symbolic = %require_complete.loc5_29 (constants.%require_complete.cbc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.loc5.%Base.loc5_22 (%Base.370), %t.param: @F.loc5.%ptr.loc5_38.1 (%ptr.b7c)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc10(%self.param: %D1, %t.param: %ptr.184) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr.573
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc5(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base.loc5_22 => constants.%Base.370
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.9f7
// CHECK:STDOUT:   %ptr.loc5_38.1 => constants.%ptr.b7c
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.8d4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%require_complete.97d
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%require_complete.cbc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn.210
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr.bfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc5(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base.loc5_22 => constants.%Base.ea5
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.3bf
// CHECK:STDOUT:   %ptr.loc5_38.1 => constants.%ptr.184
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.04d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%complete_type.513
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%complete_type.093
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_generic_specifically_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %ptr.87b: type = ptr_type %T1 [concrete]
// CHECK:STDOUT:   %pattern_type.a36: type = pattern_type %ptr.87b [concrete]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc8, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc8(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.573: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %D1: type = class_type @D1 [concrete]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F.loc8, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3bf: type = pattern_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %F.specific_fn.210: <specific function> = specific_function %F.d25, @F.loc8(%T1) [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.bfe: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T1) [concrete]
// CHECK:STDOUT:   %D1.elem: type = unbound_element_type %D1, %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.550: type = pattern_type %D1 [concrete]
// CHECK:STDOUT:   %ptr.63e: type = ptr_type %T2 [concrete]
// CHECK:STDOUT:   %pattern_type.fb8: type = pattern_type %ptr.63e [concrete]
// CHECK:STDOUT:   %F.type.af3: type = fn_type @F.loc20 [concrete]
// CHECK:STDOUT:   %F.87b: %F.type.af3 = struct_value () [concrete]
// CHECK:STDOUT:   %D1.vtable_ptr: ref %ptr.454 = vtable_ptr @D1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.fda: type = struct_type {.base: %Base.ea5} [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %struct_type.base.fda [concrete]
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
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .T2 = %T2.decl
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .D1 = %D1.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %T2.decl: type = class_decl @T2 [concrete = constants.%T2] {} {}
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D1.decl: type = class_decl @D1 [concrete = constants.%D1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T.loc7_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc8, @Base(%T.loc7_17.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2: <specific function> = specific_function %F, @F.loc8(%T.loc7_17.2) [symbolic = %F.specific_fn.loc9_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc7_17.2) [symbolic = %vtable_ptr.loc9_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc8 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc8.%pattern_type (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc8.%pattern_type (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: %pattern_type.a36 = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: %pattern_type.a36 = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc8.%Base (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc8_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc8_22.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_22.2 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc8.%Base (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: %ptr.87b = value_param call_param1
// CHECK:STDOUT:       %.loc8_33: type = splice_block %ptr [concrete = constants.%ptr.87b] {
// CHECK:STDOUT:         %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:         %ptr: type = ptr_type %T1.ref [concrete = constants.%ptr.87b]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: %ptr.87b = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc9_1.1: <specific function> = specific_function %F.decl, @F.loc8(constants.%T) [symbolic = %F.specific_fn.loc9_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc9_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc9_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc9_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .T2 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:   %.loc12: %D1.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.af3 = fn_decl @F.loc20 [concrete = constants.%F.87b] {
// CHECK:STDOUT:     %self.patt: %pattern_type.550 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.550 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.fb8 = binding_pattern t [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.fb8 = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D1 [concrete = constants.%D1]
// CHECK:STDOUT:     %self: %D1 = bind_name self, %self.param
// CHECK:STDOUT:     %t.param: %ptr.63e = value_param call_param1
// CHECK:STDOUT:     %.loc20: type = splice_block %ptr [concrete = constants.%ptr.63e] {
// CHECK:STDOUT:       %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:       %ptr: type = ptr_type %T2.ref [concrete = constants.%ptr.63e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %ptr.63e = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @D1.vtable [concrete = constants.%D1.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.ea5} [concrete = constants.%struct_type.base.fda]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.65a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc9_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @D1.vtable {
// CHECK:STDOUT:   @D1.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F.loc8(@Base.%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.9f7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.loc8.%Base (%Base.370), %t.param: %ptr.87b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc20(%self.param: %D1, %t.param: %ptr.63e) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2 => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2 => constants.%Base.vtable_ptr.573
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc8(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %F.specific_fn.loc9_1.2 => constants.%F.specific_fn.210
// CHECK:STDOUT:   %vtable_ptr.loc9_1.2 => constants.%Base.vtable_ptr.bfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc8(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.ea5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_generic_generic_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc5, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc5(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %Derived [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type.e3f: type = fn_type @F.loc16, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %F.8d6: %F.type.e3f = struct_value () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.2e6: <specific function> = specific_function %F.8d6, @F.loc16(%T) [symbolic]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.base.8ea: type = struct_type {.base: %Base} [symbolic]
// CHECK:STDOUT:   %complete_type.d5d: <witness> = complete_type_witness %struct_type.base.8ea [symbolic]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc5, @Base(%T.loc4_21.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F.loc5(%T.loc4_21.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc4_21.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc5 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.afe) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.afe) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc5.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc5.%Base (%Base) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc5.%ptr.loc5_32.1 (%ptr.79f) = value_param call_param1
// CHECK:STDOUT:       %.loc5_32: type = splice_block %ptr.loc5_32.2 [symbolic = %ptr.loc5_32.1 (constants.%ptr.79f)] {
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Base.%T.loc4_21.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %ptr.loc5_32.2: type = ptr_type %T.ref [symbolic = %ptr.loc5_32.1 (constants.%ptr.79f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @F.loc5.%ptr.loc5_32.1 (%ptr.79f) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F.loc5(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.loc7_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc8_22.2: type = class_type @Base, @Base(%T.loc7_15.2) [symbolic = %Base.loc8_22.2 (constants.%Base)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.loc8_22.2 [symbolic = %require_complete (constants.%require_complete.97d)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc7_15.2) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.loc8_22.2 [symbolic = %Derived.elem (constants.%Derived.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc16, @Derived(%T.loc7_15.2) [symbolic = %F.type (constants.%F.type.e3f)]
// CHECK:STDOUT:   %F: @Derived.%F.type (%F.type.e3f) = struct_value () [symbolic = %F (constants.%F.8d6)]
// CHECK:STDOUT:   %F.specific_fn.loc17_1.2: <specific function> = specific_function %F, @F.loc16(%T.loc7_15.2) [symbolic = %F.specific_fn.loc17_1.2 (constants.%F.specific_fn.2e6)]
// CHECK:STDOUT:   %vtable_ptr.loc17_1.2: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(%T.loc7_15.2) [symbolic = %vtable_ptr.loc17_1.2 (constants.%Derived.vtable_ptr)]
// CHECK:STDOUT:   %struct_type.base.loc17_1.2: type = struct_type {.base: @Derived.%Base.loc8_22.2 (%Base)} [symbolic = %struct_type.base.loc17_1.2 (constants.%struct_type.base.8ea)]
// CHECK:STDOUT:   %complete_type.loc17_1.2: <witness> = complete_type_witness %struct_type.base.loc17_1.2 [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.d5d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:     %Base.loc8_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc8_22.2 (constants.%Base)]
// CHECK:STDOUT:     %.loc8: @Derived.%Derived.elem (%Derived.elem) = base_decl %Base.loc8_22.1, element0 [concrete]
// CHECK:STDOUT:     %F.decl: @Derived.%F.type (%F.type.e3f) = fn_decl @F.loc16 [symbolic = @Derived.%F (constants.%F.8d6)] {
// CHECK:STDOUT:       %self.patt: @F.loc16.%pattern_type.loc16_13 (%pattern_type.423) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc16.%pattern_type.loc16_13 (%pattern_type.423) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc16.%pattern_type.loc16_25 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc16.%pattern_type.loc16_25 (%pattern_type.7dc) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc16.%Derived (%Derived) = value_param call_param0
// CHECK:STDOUT:       %.loc16_19.1: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived)] {
// CHECK:STDOUT:         %.loc16_19.2: type = specific_constant constants.%Derived, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc16_19.2 [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc16.%Derived (%Derived) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc16.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Derived.%T.loc7_15.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.loc16.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc17_1.1: <specific function> = specific_function %F.decl, @F.loc16(constants.%T) [symbolic = %F.specific_fn.loc17_1.2 (constants.%F.specific_fn.2e6)]
// CHECK:STDOUT:     %vtable_ptr.loc17_1.1: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(constants.%T) [symbolic = %vtable_ptr.loc17_1.2 (constants.%Derived.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.base.loc17_1.1: type = struct_type {.base: %Base} [symbolic = %struct_type.base.loc17_1.2 (constants.%struct_type.base.8ea)]
// CHECK:STDOUT:     %complete_type.loc17_1.1: <witness> = complete_type_witness %struct_type.base.loc17_1.1 [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.d5d)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc17_1.1
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc17_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     extend %Base.loc8_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.specific_fn.loc17_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F.loc5(@Base.%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Base [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.9f7)]
// CHECK:STDOUT:   %ptr.loc5_32.1: type = ptr_type %T [symbolic = %ptr.loc5_32.1 (constants.%ptr.79f)]
// CHECK:STDOUT:   %pattern_type.loc5_28: type = pattern_type %ptr.loc5_32.1 [symbolic = %pattern_type.loc5_28 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20: <witness> = require_complete_type %Base [symbolic = %require_complete.loc5_20 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %require_complete.loc5_29: <witness> = require_complete_type %ptr.loc5_32.1 [symbolic = %require_complete.loc5_29 (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.loc5.%Base (%Base), %t.param: @F.loc5.%ptr.loc5_32.1 (%ptr.79f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl fn @F.loc16(@Derived.%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %pattern_type.loc16_13: type = pattern_type %Derived [symbolic = %pattern_type.loc16_13 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %require_complete.loc16_28: <witness> = require_complete_type %Base [symbolic = %require_complete.loc16_28 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %pattern_type.loc16_25: type = pattern_type %T [symbolic = %pattern_type.loc16_25 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_17: <witness> = require_complete_type %Derived [symbolic = %require_complete.loc16_17 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %require_complete.loc16_26: <witness> = require_complete_type %T [symbolic = %require_complete.loc16_26 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl fn(%self.param: @F.loc16.%Derived (%Derived), %t.param: @F.loc16.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc5(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.9f7
// CHECK:STDOUT:   %ptr.loc5_32.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%require_complete.97d
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc8_22.2 => constants.%Base
// CHECK:STDOUT:   %require_complete => constants.%require_complete.97d
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.elem => constants.%Derived.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.e3f
// CHECK:STDOUT:   %F => constants.%F.8d6
// CHECK:STDOUT:   %F.specific_fn.loc17_1.2 => constants.%F.specific_fn.2e6
// CHECK:STDOUT:   %vtable_ptr.loc17_1.2 => constants.%Derived.vtable_ptr
// CHECK:STDOUT:   %struct_type.base.loc17_1.2 => constants.%struct_type.base.8ea
// CHECK:STDOUT:   %complete_type.loc17_1.2 => constants.%complete_type.d5d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc16(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %pattern_type.loc16_13 => constants.%pattern_type.423
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %require_complete.loc16_28 => constants.%require_complete.97d
// CHECK:STDOUT:   %pattern_type.loc16_25 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_17 => constants.%require_complete.5f4
// CHECK:STDOUT:   %require_complete.loc16_26 => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc5, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc5(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.573: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.97d: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Base.16b: type = class_type @Base, @Base(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %F.type.5df: type = fn_type @F.loc5, @Base(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %F.ad0: %F.type.5df = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.4a0: type = pattern_type %Base.16b [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %F.specific_fn.494: <specific function> = specific_function %F.ad0, @F.loc5(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.f98: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%ptr.79f) [symbolic]
// CHECK:STDOUT:   %require_complete.fce: <witness> = require_complete_type %Base.16b [symbolic]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.16b [symbolic]
// CHECK:STDOUT:   %pattern_type.423: type = pattern_type %Derived [symbolic]
// CHECK:STDOUT:   %F.type.e3f: type = fn_type @F.loc9, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %F.8d6: %F.type.e3f = struct_value () [symbolic]
// CHECK:STDOUT:   %F.specific_fn.2e6: <specific function> = specific_function %F.8d6, @F.loc9(%T) [symbolic]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.base.d96: type = struct_type {.base: %Base.16b} [symbolic]
// CHECK:STDOUT:   %complete_type.5dc: <witness> = complete_type_witness %struct_type.base.d96 [symbolic]
// CHECK:STDOUT:   %require_complete.5f4: <witness> = require_complete_type %Derived [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T.loc4_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_21.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc5, @Base(%T.loc4_21.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F.loc5(%T.loc4_21.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc4_21.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc5 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc5.%pattern_type.loc5_16 (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.7dc) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc5.%pattern_type.loc5_28 (%pattern_type.7dc) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc5.%Base (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc5.%Base (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc5.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Base.%T.loc4_21.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.loc5.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F.loc5(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T.loc7_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc8_22.2: type = ptr_type %T.loc7_15.2 [symbolic = %ptr.loc8_22.2 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Base.loc8_23.2: type = class_type @Base, @Base(%ptr.loc8_22.2) [symbolic = %Base.loc8_23.2 (constants.%Base.16b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.loc8_23.2 [symbolic = %require_complete (constants.%require_complete.fce)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc7_15.2) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.loc8_23.2 [symbolic = %Derived.elem (constants.%Derived.elem)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc9, @Derived(%T.loc7_15.2) [symbolic = %F.type (constants.%F.type.e3f)]
// CHECK:STDOUT:   %F: @Derived.%F.type (%F.type.e3f) = struct_value () [symbolic = %F (constants.%F.8d6)]
// CHECK:STDOUT:   %F.specific_fn.loc10_1.2: <specific function> = specific_function %F, @F.loc9(%T.loc7_15.2) [symbolic = %F.specific_fn.loc10_1.2 (constants.%F.specific_fn.2e6)]
// CHECK:STDOUT:   %vtable_ptr.loc10_1.2: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(%T.loc7_15.2) [symbolic = %vtable_ptr.loc10_1.2 (constants.%Derived.vtable_ptr)]
// CHECK:STDOUT:   %struct_type.base.loc10_1.2: type = struct_type {.base: @Derived.%Base.loc8_23.2 (%Base.16b)} [symbolic = %struct_type.base.loc10_1.2 (constants.%struct_type.base.d96)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base.loc10_1.2 [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.5dc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_15.1 [symbolic = %T.loc7_15.2 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc8_22.1: type = ptr_type %T.ref [symbolic = %ptr.loc8_22.2 (constants.%ptr.79f)]
// CHECK:STDOUT:     %Base.loc8_23.1: type = class_type @Base, @Base(constants.%ptr.79f) [symbolic = %Base.loc8_23.2 (constants.%Base.16b)]
// CHECK:STDOUT:     %.loc8: @Derived.%Derived.elem (%Derived.elem) = base_decl %Base.loc8_23.1, element0 [concrete]
// CHECK:STDOUT:     %F.decl: @Derived.%F.type (%F.type.e3f) = fn_decl @F.loc9 [symbolic = @Derived.%F (constants.%F.8d6)] {
// CHECK:STDOUT:       %self.patt: @F.loc9.%pattern_type.loc9_13 (%pattern_type.423) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc9.%pattern_type.loc9_13 (%pattern_type.423) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @F.loc9.%pattern_type.loc9_25 (%pattern_type.afe) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @F.loc9.%pattern_type.loc9_25 (%pattern_type.afe) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc9.%Derived (%Derived) = value_param call_param0
// CHECK:STDOUT:       %.loc9_19.1: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived)] {
// CHECK:STDOUT:         %.loc9_19.2: type = specific_constant constants.%Derived, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_19.2 [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc9.%Derived (%Derived) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.loc9.%ptr.loc9_28 (%ptr.79f) = value_param call_param1
// CHECK:STDOUT:       %.loc9_29: type = splice_block %ptr.loc9_29 [symbolic = %ptr.loc9_28 (constants.%ptr.79f)] {
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Derived.%T.loc7_15.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %ptr.loc9_29: type = ptr_type %T.ref [symbolic = %ptr.loc9_28 (constants.%ptr.79f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @F.loc9.%ptr.loc9_28 (%ptr.79f) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc10_1.1: <specific function> = specific_function %F.decl, @F.loc9(constants.%T) [symbolic = %F.specific_fn.loc10_1.2 (constants.%F.specific_fn.2e6)]
// CHECK:STDOUT:     %vtable_ptr.loc10_1.1: ref %ptr.454 = vtable_ptr @Derived.vtable, @Derived(constants.%T) [symbolic = %vtable_ptr.loc10_1.2 (constants.%Derived.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.base.loc10_1.1: type = struct_type {.base: %Base.16b} [symbolic = %struct_type.base.loc10_1.2 (constants.%struct_type.base.d96)]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness %struct_type.base.loc10_1.1 [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.5dc)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     extend %Base.loc8_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.specific_fn.loc10_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F.loc5(@Base.%T.loc4_21.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type.loc5_16: type = pattern_type %Base [symbolic = %pattern_type.loc5_16 (constants.%pattern_type.9f7)]
// CHECK:STDOUT:   %pattern_type.loc5_28: type = pattern_type %T [symbolic = %pattern_type.loc5_28 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20: <witness> = require_complete_type %Base [symbolic = %require_complete.loc5_20 (constants.%require_complete.97d)]
// CHECK:STDOUT:   %require_complete.loc5_29: <witness> = require_complete_type %T [symbolic = %require_complete.loc5_29 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.loc5.%Base (%Base.370), %t.param: @F.loc5.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl fn @F.loc9(@Derived.%T.loc7_15.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %pattern_type.loc9_13: type = pattern_type %Derived [symbolic = %pattern_type.loc9_13 (constants.%pattern_type.423)]
// CHECK:STDOUT:   %ptr.loc9_28: type = ptr_type %T [symbolic = %ptr.loc9_28 (constants.%ptr.79f)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%ptr.loc9_28) [symbolic = %Base (constants.%Base.16b)]
// CHECK:STDOUT:   %require_complete.loc9_28: <witness> = require_complete_type %Base [symbolic = %require_complete.loc9_28 (constants.%require_complete.fce)]
// CHECK:STDOUT:   %pattern_type.loc9_25: type = pattern_type %ptr.loc9_28 [symbolic = %pattern_type.loc9_25 (constants.%pattern_type.afe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_17: <witness> = require_complete_type %Derived [symbolic = %require_complete.loc9_17 (constants.%require_complete.5f4)]
// CHECK:STDOUT:   %require_complete.loc9_26: <witness> = require_complete_type %ptr.loc9_28 [symbolic = %require_complete.loc9_26 (constants.%require_complete.6e5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl fn(%self.param: @F.loc9.%Derived (%Derived), %t.param: @F.loc9.%ptr.loc9_28 (%ptr.79f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr.573
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc5(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.9f7
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%require_complete.97d
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc7_15.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc8_22.2 => constants.%ptr.79f
// CHECK:STDOUT:   %Base.loc8_23.2 => constants.%Base.16b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.fce
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.elem => constants.%Derived.elem
// CHECK:STDOUT:   %F.type => constants.%F.type.e3f
// CHECK:STDOUT:   %F => constants.%F.8d6
// CHECK:STDOUT:   %F.specific_fn.loc10_1.2 => constants.%F.specific_fn.2e6
// CHECK:STDOUT:   %vtable_ptr.loc10_1.2 => constants.%Derived.vtable_ptr
// CHECK:STDOUT:   %struct_type.base.loc10_1.2 => constants.%struct_type.base.d96
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.5dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%ptr.79f) {
// CHECK:STDOUT:   %T.loc4_21.2 => constants.%ptr.79f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.5df
// CHECK:STDOUT:   %F => constants.%F.ad0
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn.494
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr.f98
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc5(constants.%ptr.79f) {
// CHECK:STDOUT:   %T => constants.%ptr.79f
// CHECK:STDOUT:   %Base => constants.%Base.16b
// CHECK:STDOUT:   %pattern_type.loc5_16 => constants.%pattern_type.4a0
// CHECK:STDOUT:   %pattern_type.loc5_28 => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_20 => constants.%require_complete.fce
// CHECK:STDOUT:   %require_complete.loc5_29 => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc9(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %pattern_type.loc9_13 => constants.%pattern_type.423
// CHECK:STDOUT:   %ptr.loc9_28 => constants.%ptr.79f
// CHECK:STDOUT:   %Base => constants.%Base.16b
// CHECK:STDOUT:   %require_complete.loc9_28 => constants.%require_complete.fce
// CHECK:STDOUT:   %pattern_type.loc9_25 => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_17 => constants.%require_complete.5f4
// CHECK:STDOUT:   %require_complete.loc9_26 => constants.%require_complete.6e5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_abstract_generic_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.loc19, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F.loc19(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.573: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F.loc19, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3bf: type = pattern_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %F.specific_fn.210: <specific function> = specific_function %F.d25, @F.loc19(%T1) [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.bfe: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.ea5 [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.loc26 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.fda: type = struct_type {.base: %Base.ea5} [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %struct_type.base.fda [concrete]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_21.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_21.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_21.1: type) {
// CHECK:STDOUT:   %T.loc7_21.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_21.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.loc19, @Base(%T.loc7_21.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn.loc20_1.2: <specific function> = specific_function %F, @F.loc19(%T.loc7_21.2) [symbolic = %F.specific_fn.loc20_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr.loc20_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc7_21.2) [symbolic = %vtable_ptr.loc20_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.loc19 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.loc19.%pattern_type (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.loc19.%pattern_type (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.loc19.%Base (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc19_23.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc19_23.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc19_23.2 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.loc19.%Base (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc20_1.1: <specific function> = specific_function %F.decl, @F.loc19(constants.%T) [symbolic = %F.specific_fn.loc20_1.2 (constants.%F.specific_fn.892)]
// CHECK:STDOUT:     %vtable_ptr.loc20_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc20_1.2 (constants.%Base.vtable_ptr.573)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc20_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:   %.loc25: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.loc26 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fb9 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fb9 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.ea5} [concrete = constants.%struct_type.base.fda]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.65a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_ptr = %vtable_ptr
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc25
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc20_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic abstract fn @F.loc19(@Base.%T.loc7_21.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.9f7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   abstract fn(%self.param: @F.loc19.%Base (%Base.370));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.loc26(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_21.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc19(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_21.2 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %F.specific_fn.loc20_1.2 => constants.%F.specific_fn.210
// CHECK:STDOUT:   %vtable_ptr.loc20_1.2 => constants.%Base.vtable_ptr.bfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.loc19(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.ea5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Base(%T.loc4_17.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2: <specific function> = specific_function %F, @F(%T.loc4_17.2) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T.loc4_17.2) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type) = fn_decl @F [symbolic = @Base.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%pattern_type (%pattern_type.9f7) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @F.%pattern_type (%pattern_type.9f7) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc5_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc5_22.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_22.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.%Base (%Base) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %F.specific_fn.loc6_1.1: <specific function> = specific_function %F.decl, @F(constants.%T) [symbolic = %F.specific_fn.loc6_1.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %vtable_ptr.loc6_1.1: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T) [symbolic = %vtable_ptr.loc6_1.2 (constants.%Base.vtable_ptr)]
// CHECK:STDOUT:     %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_ptr = %vtable_ptr.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @F(@Base.%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.9f7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @F.%Base (%Base)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %F.specific_fn.loc6_1.2 => constants.%F.specific_fn
// CHECK:STDOUT:   %vtable_ptr.loc6_1.2 => constants.%Base.vtable_ptr
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.9f7: type = pattern_type %Base.370 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.892: <specific function> = specific_function %F.e26, @F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.f4433b.1: ref %ptr.454 = vtable_ptr @Base.vtable.1, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.370 [symbolic]
// CHECK:STDOUT:   %Base.vtable_ptr.f4433b.2: ref %ptr.454 = vtable_ptr @Base.vtable.2, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.3bf: type = pattern_type %Base.ea5 [concrete]
// CHECK:STDOUT:   %F.specific_fn.210: <specific function> = specific_function %F.d25, @F(%T1) [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.5be: ref %ptr.454 = vtable_ptr @Base.vtable.1, @Base(%T1) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr.f4433b.3: ref %ptr.454 = vtable_ptr @Base.vtable.3, @Base(%T) [symbolic]
// CHECK:STDOUT:   %.b74: ref %ptr.454 = class_element_access file.%v.var, element0 [concrete]
// CHECK:STDOUT:   %Base.val: %Base.ea5 = struct_value (%Base.vtable_ptr.5be) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base: %Base.type = import_ref Main//generic_lib, Base, loaded [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.03f: ref %ptr.454 = import_ref Main//generic_lib, loc6_1, loaded [symbolic = @Base.%vtable_ptr (constants.%Base.vtable_ptr.f4433b.3)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//generic_lib, loc4_17, loaded [symbolic = @Base.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.05e: <witness> = import_ref Main//generic_lib, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.8e0 = import_ref Main//generic_lib, inst26 [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.e54 = import_ref Main//generic_lib, loc5_30, unloaded
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//generic_lib, loc4_17, loaded [symbolic = @Base.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = imports.%Main.Base
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .T1 = %T1.decl
// CHECK:STDOUT:     .v = %v
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.3bf = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.3bf = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base.ea5 = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %Base [concrete = constants.%Base.ea5] {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, imports.%Main.Base [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, %T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.ea5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base.ea5 = bind_name v, %v.var [concrete = %v.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(imports.%Main.import_ref.5ab3ec.1: type) [from "generic_lib.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Base(%T) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic = %F.specific_fn (constants.%F.specific_fn.892)]
// CHECK:STDOUT:   %vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable.1, @Base(%T) [symbolic = %vtable_ptr (constants.%Base.vtable_ptr.f4433b.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.05e
// CHECK:STDOUT:     vtable_ptr = imports.%Main.import_ref.03f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.8e0
// CHECK:STDOUT:     .F = imports.%Main.import_ref.e54
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable.1 {
// CHECK:STDOUT:   constants.%F.specific_fn.892
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable.2 {
// CHECK:STDOUT:   constants.%F.specific_fn.892
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable.3 {
// CHECK:STDOUT:   constants.%F.specific_fn.892
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(imports.%Main.import_ref.5ab3ec.2: type) [from "generic_lib.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.9f7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_20.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_20.2: ref %ptr.454 = class_element_access file.%v.var, element0 [concrete = constants.%.b74]
// CHECK:STDOUT:   %.loc7_20.3: init %ptr.454 = initialize_from constants.%Base.vtable_ptr.5be to %.loc7_20.2 [concrete = constants.%Base.vtable_ptr.5be]
// CHECK:STDOUT:   %.loc7_20.4: init %Base.ea5 = class_init (%.loc7_20.3), file.%v.var [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_1: init %Base.ea5 = converted %.loc7_20.1, %.loc7_20.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign file.%v.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.f17
// CHECK:STDOUT:   %F => constants.%F.e26
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.892
// CHECK:STDOUT:   %vtable_ptr => constants.%Base.vtable_ptr.f4433b.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %F.specific_fn => constants.%F.specific_fn.210
// CHECK:STDOUT:   %vtable_ptr => constants.%Base.vtable_ptr.5be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.ea5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3bf
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
