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
  override fn H[self: Self]();
}

fn Use() {
  var unused d: Derived = {.base = {}};
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
  var unused v: Modifiers.Base = {};
}

// --- impl_abstract.carbon

library "[[@TEST_NAME]]";

abstract class A1 {
  virtual fn F[self: Self]();
}

abstract class A2 {
  extend base: A1;
  override fn F[self: Self]();
}

// --- impl_base.carbon

library "[[@TEST_NAME]]";

base class B1 {
  virtual fn F[self: Self]();
}

base class B2 {
  extend base: B1;
  override fn F[self: Self]();
}

class C {
  extend base: B2;
  override fn F[self: Self]();
}

fn Use() {
  var unused b1: B1 = {};
  var unused b2: B2 = {.base = {}};
  var unused c: C = {.base = {.base = {}}};
}

// --- fail_modifiers.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:3: error: override without base class [OverrideWithoutBase]
  // CHECK:STDERR:   override fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[self: Self]();
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
  var b1: Base = {.m2 = i, .m1 = i};
  var unused b2: Base = {.m2 = 3, .m1 = 5};

  b1.m2 = 4;
}

// --- fail_impl_without_base_declaration.carbon

library "[[@TEST_NAME]]";

base class Base {
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_impl_without_base_declaration.carbon:[[@LINE+4]]:3: error: override without compatible virtual in base class [OverrideWithoutVirtualInBase]
  // CHECK:STDERR:   override fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[self: Self]();
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
  override fn F[self: Self]();
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
  override fn F[self: Self]();
}

// --- fail_impl_mismatch.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE+7]]:3: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
  // CHECK:STDERR:   override fn F[self: Self](v: i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE-8]]:3: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[self: Self](v: i32);
}

// --- fail_todo_impl_conversion.carbon

library "[[@TEST_NAME]]";

class T1 {
}

class T2 {
}

impl T2 as Core.ImplicitAs(T1) {
  fn Convert[unused self: Self]() -> T1 {
    return {};
  }
}

base class Base {
  virtual fn F[self: Self]() -> T1;
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_todo_impl_conversion.carbon:[[@LINE+7]]:3: error: function redeclaration differs because return type is `T2` [FunctionRedeclReturnTypeDiffers]
  // CHECK:STDERR:   override fn F[self: Self]() -> T2;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_todo_impl_conversion.carbon:[[@LINE-8]]:3: note: previously declared with return type `T1` [FunctionRedeclReturnTypePrevious]
  // CHECK:STDERR:   virtual fn F[self: Self]() -> T1;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[self: Self]() -> T2;
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
  virtual fn F[unused self: Self](unused t: T) { }
}

class Derived {
  extend base: Base(T1);
  override fn F[unused self: Self](unused t: T1) { }
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
  // CHECK:STDERR:   override fn F();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F();
}

// --- fail_ref_self_mismatch.carbon

library "[[@TEST_NAME]]";

base class T1 {
  virtual fn F1[self: Self]();
}

class T2 {
  extend base: T1;
  // CHECK:STDERR: fail_ref_self_mismatch.carbon:[[@LINE+7]]:18: error: redeclaration differs at implicit parameter 1 [RedeclParamDiffers]
  // CHECK:STDERR:   override fn F1[ref self: Self]();
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_ref_self_mismatch.carbon:[[@LINE-8]]:17: note: previous declaration's corresponding implicit parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F1[self: Self]();
  // CHECK:STDERR:                 ^~~~~~~~~~
  // CHECK:STDERR:
  override fn F1[ref self: Self]();
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
  virtual fn F[unused self: Self]() { }
}

// --- with_dependent_arg.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  virtual fn F[unused self: Self](unused t: T) { }
}

// --- vtable_import_unneeded.carbon

library "[[@TEST_NAME]]";

import Modifiers;

fn F(b: Modifiers.Base);

// --- generic_derived_from_nongeneric.carbon

library "[[@TEST_NAME]]";

base class NonGenericBase {
  virtual fn F1[unused self: Self]() { }
  virtual fn F2[unused self: Self]() { }
}

base class GenericDerived(T:! type) {
  extend base: NonGenericBase;
  override fn F2[unused self: Self]() { }
  virtual fn F3[unused self: Self]() { }
}

// --- nongeneric_derived_from_generic.carbon

library "[[@TEST_NAME]]";

base class GenericBase(T:! type) {
  virtual fn F1[unused self: Self]() { }
  virtual fn F2[unused self: Self]() { }
}

class T1;

base class NonGenericDerived {
  extend base: GenericBase(T1);
  override fn F2[unused self: Self]() { }
  virtual fn F3[unused self: Self]() { }
}

// --- impl_generic_specifically.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[unused self: Self](unused t: Base(T)*) { }
}
class T1;
class D1 {
  extend base: Base(T1);
  override fn F[unused self: Self](unused t: Base(T1)*) { }
}

// --- fail_impl_generic_specifically_mismatch.carbon

library "[[@TEST_NAME]]";

class T1;
class T2;

base class Base(T:! type) {
  virtual fn F[unused self: Self](unused t: T1*) { }
}

class D1 {
  extend base: Base(T1);
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE+7]]:43: error: type `<pattern for T2*>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T1*>` [RedeclParamDiffersType]
  // CHECK:STDERR:   override fn F[unused self: Self](unused t: T2*) { }
  // CHECK:STDERR:                                           ^~~~~~
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE-8]]:42: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T1*) { }
  // CHECK:STDERR:                                          ^~~~~~
  // CHECK:STDERR:
  override fn F[unused self: Self](unused t: T2*) { }
}

// --- fail_impl_generic_generic_mismatch.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  virtual fn F[unused self: Self](unused t: T*) { }
}
class Derived(T:! type) {
  extend base: Base(T);
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE+7]]:43: error: type `<pattern for T>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T*>` [RedeclParamDiffersType]
  // CHECK:STDERR:   override fn F[unused self: Self](unused t: T) { }
  // CHECK:STDERR:                                           ^~~~
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE-7]]:42: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T*) { }
  // CHECK:STDERR:                                          ^~~~~
  // CHECK:STDERR:
  override fn F[unused self: Self](unused t: T) { }
}

// --- impl_generic_generic.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  virtual fn F[unused self: Self](unused t: T) { }
}
class Derived(T:! type) {
  extend base: Base(T*);
  override fn F[unused self: Self](unused t: T*) { }
}

// --- abstract_generic_undefined.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  abstract fn F[self: Self]();
}

class T1;

class Derived {
  extend base: Base(T1);
  override fn F[unused self: Self]() { }
}

// --- generic_lib.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  virtual fn F[unused self: Self]() { }
}

// --- generic_import.carbon

library "[[@TEST_NAME]]";

import library "generic_lib";

class T1;
var v: Base(T1) = {};

// --- generic_derived_generic.carbon

library "[[@TEST_NAME]]";

base class T1(G1:! type) {
  virtual fn F[unused self: Self]() { }
}

class T2(G2:! type) {
  extend base: T1(G2);
}

// --- generic_derived_generic_context.carbon

library "[[@TEST_NAME]]";

base class T1(G1:! type) {
  virtual fn F[unused self: Self]() { }
}

class T2(G2:! type) {
  class T3 {
    extend base : T1(G2);
  }
}


// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.dec: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.3c9: type = pattern_type %Abstract [concrete]
// CHECK:STDOUT:   %Abstract.J.type: type = fn_type @Abstract.J [concrete]
// CHECK:STDOUT:   %Abstract.J: %Abstract.J.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract.K.type: type = fn_type @Abstract.K [concrete]
// CHECK:STDOUT:   %Abstract.K: %Abstract.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract.vtable_decl: ref %ptr = vtable_decl @Abstract.vtable [concrete]
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
// CHECK:STDOUT:   %Base.H.decl: %Base.H.type = fn_decl @Base.H [concrete = constants.%Base.H] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dec = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.dec = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %Base.H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %Abstract.J.decl: %Abstract.J.type = fn_decl @Abstract.J [concrete = constants.%Abstract.J] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3c9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.3c9 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.K.decl: %Abstract.K.type = fn_decl @Abstract.K [concrete = constants.%Abstract.K] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.3c9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.3c9 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr = vtable_decl @Abstract.vtable [concrete = constants.%Abstract.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %Abstract.J.decl
// CHECK:STDOUT:   .K = %Abstract.K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Abstract.vtable {
// CHECK:STDOUT:   @Abstract.%Abstract.J.decl
// CHECK:STDOUT:   @Abstract.%Abstract.K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.H(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @Abstract.J(%self.param: %Abstract);
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Abstract.K(%self.param: %Abstract);
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
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.H.type: type = fn_type @Derived.H [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.H: %Derived.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.96c: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base.96c [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.6b1: %struct_type.base.f5e = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %.a5d: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %uninit: %ptr.454 = uninitialized_value [concrete]
// CHECK:STDOUT:   %struct.4dc: %.a5d = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.5: type = fn_type @Destroy.Op.loc12_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.5: %Destroy.Op.type.bae255.5 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Modifiers.import_ref.695: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   %Modifiers.import_ref.d82 = import_ref Modifiers//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.d40 = import_ref Modifiers//default, loc5_29, unloaded
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
// CHECK:STDOUT:   %Derived.H.decl: %Derived.H.type = fn_decl @Derived.H [concrete = constants.%Derived.H] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.96c [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Modifiers = <poisoned>
// CHECK:STDOUT:   .base = %.loc7
// CHECK:STDOUT:   .H = %Derived.H.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_decl = imports.%Modifiers.import_ref.695
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.d82
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.d40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.H.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.H(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.H [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.9f6 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.9f6 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %Derived = var %d.var_patt
// CHECK:STDOUT:   %.loc12_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_38.1: %struct_type.base.f5e = struct_literal (%.loc12_37.1) [concrete = constants.%struct.6b1]
// CHECK:STDOUT:   %.loc12_38.2: ref %.a5d = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc12_37.2: ref %ptr.454 = class_element_access %.loc12_38.2, element0
// CHECK:STDOUT:   %uninit: %ptr.454 = uninitialized_value [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc12_37.3: init %ptr.454 to %.loc12_37.2 = in_place_init %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc12_37.4: init %.a5d to %.loc12_38.2 = class_init (%.loc12_37.3) [concrete = constants.%struct.4dc]
// CHECK:STDOUT:   %.loc12_38.3: init %.a5d = converted %.loc12_37.1, %.loc12_37.4 [concrete = constants.%struct.4dc]
// CHECK:STDOUT:   %.loc12_38.4: init %Base = as_compatible %.loc12_38.3 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc12_38.5: init %Derived to %d.var = class_init (%.loc12_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %.loc12_38.6: ref %Base = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc12_38.7: ref %ptr.454 = class_element_access %.loc12_38.6, element0
// CHECK:STDOUT:   %.loc12_38.8: init %ptr.454 to %.loc12_38.7 = in_place_init %Derived.vtable_ptr [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %.loc12_38.9: init %Derived = update_init %.loc12_38.5, %.loc12_38.8
// CHECK:STDOUT:   %.loc12_3: init %Derived = converted %.loc12_38.1, %.loc12_38.9
// CHECK:STDOUT:   assign %d.var, %.loc12_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d: ref %Derived = ref_binding d, %d.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.3(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.4(%self.param: ref %struct_type.base.96c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc12_3.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
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
// CHECK:STDOUT:   %Modifiers.import_ref.695: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   %Modifiers.import_ref.d82 = import_ref Modifiers//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.d40 = import_ref Modifiers//default, loc5_29, unloaded
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
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   .Modifiers = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_decl = imports.%Modifiers.import_ref.695
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.d82
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.d40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Derived.F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.H [from "modifiers.carbon"];
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %pattern_type.80f: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc7_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Modifiers.import_ref.695: ref %ptr.454 = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Modifiers.import_ref.d82 = import_ref Modifiers//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.d40 = import_ref Modifiers//default, loc5_29, unloaded
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
// CHECK:STDOUT:   vtable_decl = imports.%Modifiers.import_ref.695
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.d82
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.d40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.80f = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.80f = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base = var %v.var_patt
// CHECK:STDOUT:   %.loc7_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_35.2: ref %ptr.454 = class_element_access %v.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc7_35.3: init %ptr.454 to %.loc7_35.2 = in_place_init %Base.vtable_ptr [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc7_35.4: init %Base to %v.var = class_init (%.loc7_35.3) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_3: init %Base = converted %.loc7_35.1, %.loc7_35.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %v.var, %.loc7_3
// CHECK:STDOUT:   %.loc7_26: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = ref_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.H [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc7_3.3(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %pattern_type.258: type = pattern_type %A1 [concrete]
// CHECK:STDOUT:   %A1.F.type: type = fn_type @A1.F [concrete]
// CHECK:STDOUT:   %A1.F: %A1.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %A1.vtable_decl: ref %ptr.454 = vtable_decl @A1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %A1 [concrete]
// CHECK:STDOUT:   %pattern_type.f23: type = pattern_type %A2 [concrete]
// CHECK:STDOUT:   %A2.F.type: type = fn_type @A2.F [concrete]
// CHECK:STDOUT:   %A2.F: %A2.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A2.vtable_decl: ref %ptr.454 = vtable_decl @A2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A1} [concrete]
// CHECK:STDOUT:   %complete_type.0a6: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:   %A1.F.decl: %A1.F.type = fn_decl @A1.F [concrete = constants.%A1.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.258 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.258 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A1 [concrete = constants.%A1]
// CHECK:STDOUT:     %self: %A1 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @A1.vtable [concrete = constants.%A1.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .F = %A1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %A1.ref: type = name_ref A1, file.%A1.decl [concrete = constants.%A1]
// CHECK:STDOUT:   %.loc9: %A2.elem = base_decl %A1.ref, element0 [concrete]
// CHECK:STDOUT:   %A2.F.decl: %A2.F.type = fn_decl @A2.F [concrete = constants.%A2.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.f23 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.f23 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A2 [concrete = constants.%A2]
// CHECK:STDOUT:     %self: %A2 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @A2.vtable [concrete = constants.%A2.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.0a6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .A1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %A2.F.decl
// CHECK:STDOUT:   extend %A1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @A1.vtable {
// CHECK:STDOUT:   @A1.%A1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @A2.vtable {
// CHECK:STDOUT:   @A2.%A2.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @A1.F(%self.param: %A1);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @A2.F(%self.param: %A2);
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.748: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %B1.F.type: type = fn_type @B1.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %B1.F: %B1.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %B1.vtable_decl: ref %ptr.454 = vtable_decl @B1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %B2.elem: type = unbound_element_type %B2, %B1 [concrete]
// CHECK:STDOUT:   %pattern_type.8b5: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %B2.F.type: type = fn_type @B2.F [concrete]
// CHECK:STDOUT:   %B2.F: %B2.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %B2.vtable_decl: ref %ptr.454 = vtable_decl @B2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.056: type = struct_type {.base: %B1} [concrete]
// CHECK:STDOUT:   %complete_type.4f8: <witness> = complete_type_witness %struct_type.base.056 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B2 [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.vtable_decl: ref %ptr.454 = vtable_decl @C.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.812: type = struct_type {.base: %B2} [concrete]
// CHECK:STDOUT:   %complete_type.909: <witness> = complete_type_witness %struct_type.base.812 [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %B1.vtable_ptr: ref %ptr.454 = vtable_ptr @B1.vtable [concrete]
// CHECK:STDOUT:   %B1.val.d17: %B1 = struct_value (%B1.vtable_ptr) [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.6b1: %struct_type.base.f5e = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %.5b3: type = partial_type %B1 [concrete]
// CHECK:STDOUT:   %uninit: %ptr.454 = uninitialized_value [concrete]
// CHECK:STDOUT:   %struct.571: %.5b3 = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %B1.val.fd5: %B1 = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %B2.val: %B2 = struct_value (%B1.val.fd5) [concrete]
// CHECK:STDOUT:   %B2.vtable_ptr: ref %ptr.454 = vtable_ptr @B2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.a0c: type = struct_type {.base: %struct_type.base.f5e} [concrete]
// CHECK:STDOUT:   %struct.83e: %struct_type.base.a0c = struct_value (%struct.6b1) [concrete]
// CHECK:STDOUT:   %.312: type = partial_type %B2 [concrete]
// CHECK:STDOUT:   %struct.0cd: %.312 = struct_value (%B1.val.fd5) [concrete]
// CHECK:STDOUT:   %C.val: %C = struct_value (%B2.val) [concrete]
// CHECK:STDOUT:   %C.vtable_ptr: ref %ptr.454 = vtable_ptr @C.vtable [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc21_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.5: type = fn_type @Destroy.Op.loc21_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.5: %Destroy.Op.type.bae255.5 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.7: type = fn_type @Destroy.Op.loc21_3.7 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.7: %Destroy.Op.type.bae255.7 = struct_value () [concrete]
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
// CHECK:STDOUT:   %B1.F.decl: %B1.F.type = fn_decl @B1.F [concrete = constants.%B1.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.748 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.748 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B1 [concrete = constants.%B1]
// CHECK:STDOUT:     %self: %B1 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @B1.vtable [concrete = constants.%B1.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B1
// CHECK:STDOUT:   .F = %B1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %.loc9: %B2.elem = base_decl %B1.ref, element0 [concrete]
// CHECK:STDOUT:   %B2.F.decl: %B2.F.type = fn_decl @B2.F [concrete = constants.%B2.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.8b5 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.8b5 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B2 [concrete = constants.%B2]
// CHECK:STDOUT:     %self: %B2 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @B2.vtable [concrete = constants.%B2.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.056 [concrete = constants.%complete_type.4f8]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .B1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %B2.F.decl
// CHECK:STDOUT:   extend %B1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B2.ref, element0 [concrete]
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7c7 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7c7 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @C.vtable [concrete = constants.%C.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.812 [concrete = constants.%complete_type.909]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B2 = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT:   extend %B2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @B1.vtable {
// CHECK:STDOUT:   @B1.%B1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @B2.vtable {
// CHECK:STDOUT:   @B2.%B2.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @C.vtable {
// CHECK:STDOUT:   @C.%C.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @B1.F(%self.param: %B1);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @B2.F(%self.param: %B2);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @C.F(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.748 = ref_binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.748 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %B1 = var %b1.var_patt
// CHECK:STDOUT:   %.loc19_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc19_24.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %B1.vtable_ptr: ref %ptr.454 = vtable_ptr @B1.vtable [concrete = constants.%B1.vtable_ptr]
// CHECK:STDOUT:   %.loc19_24.3: init %ptr.454 to %.loc19_24.2 = in_place_init %B1.vtable_ptr [concrete = constants.%B1.vtable_ptr]
// CHECK:STDOUT:   %.loc19_24.4: init %B1 to %b1.var = class_init (%.loc19_24.3) [concrete = constants.%B1.val.d17]
// CHECK:STDOUT:   %.loc19_3: init %B1 = converted %.loc19_24.1, %.loc19_24.4 [concrete = constants.%B1.val.d17]
// CHECK:STDOUT:   assign %b1.var, %.loc19_3
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %b1: ref %B1 = ref_binding b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.8b5 = ref_binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.8b5 = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %B2 = var %b2.var_patt
// CHECK:STDOUT:   %.loc20_33.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc20_34.1: %struct_type.base.f5e = struct_literal (%.loc20_33.1) [concrete = constants.%struct.6b1]
// CHECK:STDOUT:   %.loc20_34.2: ref %.5b3 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc20_33.2: ref %ptr.454 = class_element_access %.loc20_34.2, element0
// CHECK:STDOUT:   %uninit.loc20: %ptr.454 = uninitialized_value [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc20_33.3: init %ptr.454 to %.loc20_33.2 = in_place_init %uninit.loc20 [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc20_33.4: init %.5b3 to %.loc20_34.2 = class_init (%.loc20_33.3) [concrete = constants.%struct.571]
// CHECK:STDOUT:   %.loc20_34.3: init %.5b3 = converted %.loc20_33.1, %.loc20_33.4 [concrete = constants.%struct.571]
// CHECK:STDOUT:   %.loc20_34.4: init %B1 = as_compatible %.loc20_34.3 [concrete = constants.%B1.val.fd5]
// CHECK:STDOUT:   %.loc20_34.5: init %B2 to %b2.var = class_init (%.loc20_34.4) [concrete = constants.%B2.val]
// CHECK:STDOUT:   %B2.vtable_ptr: ref %ptr.454 = vtable_ptr @B2.vtable [concrete = constants.%B2.vtable_ptr]
// CHECK:STDOUT:   %.loc20_34.6: ref %B1 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc20_34.7: ref %ptr.454 = class_element_access %.loc20_34.6, element0
// CHECK:STDOUT:   %.loc20_34.8: init %ptr.454 to %.loc20_34.7 = in_place_init %B2.vtable_ptr [concrete = constants.%B2.vtable_ptr]
// CHECK:STDOUT:   %.loc20_34.9: init %B2 = update_init %.loc20_34.5, %.loc20_34.8
// CHECK:STDOUT:   %.loc20_3: init %B2 = converted %.loc20_34.1, %.loc20_34.9
// CHECK:STDOUT:   assign %b2.var, %.loc20_3
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %b2: ref %B2 = ref_binding b2, %b2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = ref_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.7c7 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc21_40.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc21_41.1: %struct_type.base.f5e = struct_literal (%.loc21_40.1) [concrete = constants.%struct.6b1]
// CHECK:STDOUT:   %.loc21_42.1: %struct_type.base.a0c = struct_literal (%.loc21_41.1) [concrete = constants.%struct.83e]
// CHECK:STDOUT:   %.loc21_42.2: ref %.312 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc21_41.2: ref %.5b3 = class_element_access %.loc21_42.2, element0
// CHECK:STDOUT:   %.loc21_40.2: ref %ptr.454 = class_element_access %.loc21_41.2, element0
// CHECK:STDOUT:   %uninit.loc21: %ptr.454 = uninitialized_value [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc21_40.3: init %ptr.454 to %.loc21_40.2 = in_place_init %uninit.loc21 [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc21_40.4: init %.5b3 to %.loc21_41.2 = class_init (%.loc21_40.3) [concrete = constants.%struct.571]
// CHECK:STDOUT:   %.loc21_41.3: init %.5b3 = converted %.loc21_40.1, %.loc21_40.4 [concrete = constants.%struct.571]
// CHECK:STDOUT:   %.loc21_41.4: init %B1 = as_compatible %.loc21_41.3 [concrete = constants.%B1.val.fd5]
// CHECK:STDOUT:   %.loc21_41.5: init %.312 to %.loc21_42.2 = class_init (%.loc21_41.4) [concrete = constants.%struct.0cd]
// CHECK:STDOUT:   %.loc21_42.3: init %.312 = converted %.loc21_41.1, %.loc21_41.5 [concrete = constants.%struct.0cd]
// CHECK:STDOUT:   %.loc21_42.4: init %B2 = as_compatible %.loc21_42.3 [concrete = constants.%B2.val]
// CHECK:STDOUT:   %.loc21_42.5: init %C to %c.var = class_init (%.loc21_42.4) [concrete = constants.%C.val]
// CHECK:STDOUT:   %C.vtable_ptr: ref %ptr.454 = vtable_ptr @C.vtable [concrete = constants.%C.vtable_ptr]
// CHECK:STDOUT:   %.loc21_42.6: ref %B2 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc21_42.7: ref %B1 = class_element_access %.loc21_42.6, element0
// CHECK:STDOUT:   %.loc21_42.8: ref %ptr.454 = class_element_access %.loc21_42.7, element0
// CHECK:STDOUT:   %.loc21_42.9: init %ptr.454 to %.loc21_42.8 = in_place_init %C.vtable_ptr [concrete = constants.%C.vtable_ptr]
// CHECK:STDOUT:   %.loc21_42.10: init %C = update_init %.loc21_42.5, %.loc21_42.9
// CHECK:STDOUT:   %.loc21_3: init %C = converted %.loc21_42.1, %.loc21_42.10
// CHECK:STDOUT:   assign %c.var, %.loc21_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = ref_binding c, %c.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %c.var, constants.%Destroy.Op.651ba6.7
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%c.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc20: <bound method> = bound_method %b2.var, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc20: init %empty_tuple.type = call %Destroy.Op.bound.loc20(%b2.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc19: <bound method> = bound_method %b1.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc19: init %empty_tuple.type = call %Destroy.Op.bound.loc19(%b1.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.3(%self.param: ref %B1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.4(%self.param: ref %struct_type.base.056) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.5(%self.param: ref %B2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.6(%self.param: ref %struct_type.base.812) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc21_3.7(%self.param: ref %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %C.F.type: type = fn_type @C.F [concrete]
// CHECK:STDOUT:   %C.F: %C.F.type = struct_value () [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %C.F.decl: %C.F.type = fn_decl @C.F [concrete = constants.%C.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %C.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @C.F(%self.param: %C);
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.101: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.6bc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.74f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e0d = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.6bc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.b37: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.545: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.b37, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.fa7: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.68c: type = struct_type {.m2: %i32, .m1: %i32} [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.5f2: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.m2.m1.5f2 = struct_value (%int_3.1ba, %int_5.64b) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.e9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr, %int_5.0f6, %int_3.822) [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5 [concrete]
// CHECK:STDOUT:   %bound_method.6d7: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.3: type = fn_type @Destroy.Op.loc14_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.3: %Destroy.Op.type.bae255.3 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.5: type = fn_type @Destroy.Op.loc14_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.5: %Destroy.Op.type.bae255.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.42d: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.4e6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.3c2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.74f = impl_witness_table (%Core.import_ref.42d), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %i32.loc5: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc5: %Base.elem = field_decl m1, element1 [concrete]
// CHECK:STDOUT:   %i32.loc6: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc6: %Base.elem = field_decl m2, element2 [concrete]
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr.m1.m2 [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .m1 = %.loc5
// CHECK:STDOUT:   .m2 = %.loc6
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.F(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = ref_binding_pattern i [concrete]
// CHECK:STDOUT:     %i.var_patt: %pattern_type.7ce = var_pattern %i.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %i32 = var %i.var_patt
// CHECK:STDOUT:   %int_3.loc12: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc12: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc12_3.1: <bound method> = bound_method %int_3.loc12, %impl.elem0.loc12 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_3.2: <bound method> = bound_method %int_3.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12: init %i32 = call %bound_method.loc12_3.2(%int_3.loc12) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12: init %i32 = converted %int_3.loc12, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc12 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %i.var, %.loc12
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %i: ref %i32 = ref_binding i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.101 = ref_binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.101 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %Base = var %b1.var_patt
// CHECK:STDOUT:   %i.ref.loc13_25: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc13_34: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc13_35.1: %struct_type.m2.m1.68c = struct_literal (%i.ref.loc13_25, %i.ref.loc13_34)
// CHECK:STDOUT:   %.loc13_35.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr.loc13: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc13_35.3: init %ptr.454 to %.loc13_35.2 = in_place_init %Base.vtable_ptr.loc13 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc13_34: %i32 = acquire_value %i.ref.loc13_34
// CHECK:STDOUT:   %impl.elem0.loc13_34: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc13_34.1: <bound method> = bound_method %.loc13_34, %impl.elem0.loc13_34
// CHECK:STDOUT:   %specific_fn.loc13_34: <specific function> = specific_function %impl.elem0.loc13_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_34.2: <bound method> = bound_method %.loc13_34, %specific_fn.loc13_34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_34: init %i32 = call %bound_method.loc13_34.2(%.loc13_34)
// CHECK:STDOUT:   %.loc13_35.4: ref %i32 = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc13_35.5: init %i32 to %.loc13_35.4 = in_place_init %Int.as.Copy.impl.Op.call.loc13_34
// CHECK:STDOUT:   %.loc13_25: %i32 = acquire_value %i.ref.loc13_25
// CHECK:STDOUT:   %impl.elem0.loc13_25: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc13_25.1: <bound method> = bound_method %.loc13_25, %impl.elem0.loc13_25
// CHECK:STDOUT:   %specific_fn.loc13_25: <specific function> = specific_function %impl.elem0.loc13_25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_25.2: <bound method> = bound_method %.loc13_25, %specific_fn.loc13_25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc13_25: init %i32 = call %bound_method.loc13_25.2(%.loc13_25)
// CHECK:STDOUT:   %.loc13_35.6: ref %i32 = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc13_35.7: init %i32 to %.loc13_35.6 = in_place_init %Int.as.Copy.impl.Op.call.loc13_25
// CHECK:STDOUT:   %.loc13_35.8: init %Base to %b1.var = class_init (%.loc13_35.3, %.loc13_35.5, %.loc13_35.7)
// CHECK:STDOUT:   %.loc13_3: init %Base = converted %.loc13_35.1, %.loc13_35.8
// CHECK:STDOUT:   assign %b1.var, %.loc13_3
// CHECK:STDOUT:   %Base.ref.loc13: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = ref_binding b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.101 = ref_binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.101 = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var %b2.var_patt
// CHECK:STDOUT:   %int_3.loc14: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc14_42.1: %struct_type.m2.m1.5f2 = struct_literal (%int_3.loc14, %int_5) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc14_42.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr.loc14: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc14_42.3: init %ptr.454 to %.loc14_42.2 = in_place_init %Base.vtable_ptr.loc14 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %impl.elem0.loc14_42.1: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc14_42.1: <bound method> = bound_method %int_5, %impl.elem0.loc14_42.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.005]
// CHECK:STDOUT:   %specific_fn.loc14_42.1: <specific function> = specific_function %impl.elem0.loc14_42.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_42.2: <bound method> = bound_method %int_5, %specific_fn.loc14_42.1 [concrete = constants.%bound_method.e9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_42.1: init %i32 = call %bound_method.loc14_42.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc14_42.4: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_42.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc14_42.5: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc14_42.6: init %i32 to %.loc14_42.5 = in_place_init %.loc14_42.4 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc14_42.2: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc14_42.3: <bound method> = bound_method %int_3.loc14, %impl.elem0.loc14_42.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.061]
// CHECK:STDOUT:   %specific_fn.loc14_42.2: <specific function> = specific_function %impl.elem0.loc14_42.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_42.4: <bound method> = bound_method %int_3.loc14, %specific_fn.loc14_42.2 [concrete = constants.%bound_method.fa7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_42.2: init %i32 = call %bound_method.loc14_42.4(%int_3.loc14) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_42.7: init %i32 = converted %int_3.loc14, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc14_42.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_42.8: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc14_42.9: init %i32 to %.loc14_42.8 = in_place_init %.loc14_42.7 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc14_42.10: init %Base to %b2.var = class_init (%.loc14_42.3, %.loc14_42.6, %.loc14_42.9) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc14_3: init %Base = converted %.loc14_42.1, %.loc14_42.10 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %b2.var, %.loc14_3
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = ref_binding b2, %b2.var
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %Base.elem = name_ref m2, @Base.%.loc6 [concrete = @Base.%.loc6]
// CHECK:STDOUT:   %.loc16_5: ref %i32 = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc16: %.545 = impl_witness_access constants.%ImplicitAs.impl_witness.6bc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.0b5]
// CHECK:STDOUT:   %bound_method.loc16_9.1: <bound method> = bound_method %int_4, %impl.elem0.loc16 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f0c]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_9.2: <bound method> = bound_method %int_4, %specific_fn.loc16 [concrete = constants.%bound_method.6d7]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16: init %i32 = call %bound_method.loc16_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc16_9: init %i32 = converted %int_4, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc16 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc16_5, %.loc16_9
// CHECK:STDOUT:   %Destroy.Op.bound.loc14: <bound method> = bound_method %b2.var, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc14: init %empty_tuple.type = call %Destroy.Op.bound.loc14(%b2.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc13: <bound method> = bound_method %b1.var, constants.%Destroy.Op.651ba6.5
// CHECK:STDOUT:   %Destroy.Op.call.loc13: init %empty_tuple.type = call %Destroy.Op.bound.loc13(%b1.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc12: <bound method> = bound_method %i.var, constants.%Destroy.Op.651ba6.3
// CHECK:STDOUT:   %Destroy.Op.call.loc12: init %empty_tuple.type = call %Destroy.Op.bound.loc12(%i.var)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.2(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.3(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.4(%self.param: ref %struct_type.vptr.m1.m2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc14_3.5(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_without_base_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr.base: type = struct_type {.<vptr>: %ptr.454, .base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.8aa: <witness> = complete_type_witness %struct_type.vptr.base [concrete]
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
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element1 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr.base [concrete = constants.%complete_type.8aa]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {}
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractBase: type = class_type @AbstractBase [concrete]
// CHECK:STDOUT:   %pattern_type.079: type = pattern_type %AbstractBase [concrete]
// CHECK:STDOUT:   %AbstractBase.F.type: type = fn_type @AbstractBase.F [concrete]
// CHECK:STDOUT:   %AbstractBase.F: %AbstractBase.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %AbstractBase.vtable_decl: ref %ptr.454 = vtable_decl @AbstractBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %AbstractIntermediate: type = class_type @AbstractIntermediate [concrete]
// CHECK:STDOUT:   %AbstractIntermediate.elem: type = unbound_element_type %AbstractIntermediate, %AbstractBase [concrete]
// CHECK:STDOUT:   %AbstractIntermediate.vtable_decl: ref %ptr.454 = vtable_decl @AbstractIntermediate.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.48d: type = struct_type {.base: %AbstractBase} [concrete]
// CHECK:STDOUT:   %complete_type.cae: <witness> = complete_type_witness %struct_type.base.48d [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %AbstractIntermediate [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.493: type = struct_type {.base: %AbstractIntermediate} [concrete]
// CHECK:STDOUT:   %complete_type.944: <witness> = complete_type_witness %struct_type.base.493 [concrete]
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
// CHECK:STDOUT:   %AbstractBase.F.decl: %AbstractBase.F.type = fn_decl @AbstractBase.F [concrete = constants.%AbstractBase.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.079 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.079 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AbstractBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractBase [concrete = constants.%AbstractBase]
// CHECK:STDOUT:     %self: %AbstractBase = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @AbstractBase.vtable [concrete = constants.%AbstractBase.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractBase
// CHECK:STDOUT:   .F = %AbstractBase.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractIntermediate {
// CHECK:STDOUT:   %AbstractBase.ref: type = name_ref AbstractBase, file.%AbstractBase.decl [concrete = constants.%AbstractBase]
// CHECK:STDOUT:   %.loc9: %AbstractIntermediate.elem = base_decl %AbstractBase.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @AbstractIntermediate.vtable [concrete = constants.%AbstractIntermediate.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.48d [concrete = constants.%complete_type.cae]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
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
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.493 [concrete = constants.%complete_type.944]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .AbstractIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %AbstractIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @AbstractBase.vtable {
// CHECK:STDOUT:   @AbstractBase.%AbstractBase.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @AbstractIntermediate.vtable {
// CHECK:STDOUT:   constants.%AbstractBase.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @AbstractBase.F(%self.param: %AbstractBase);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- virtual_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %VirtualBase: type = class_type @VirtualBase [concrete]
// CHECK:STDOUT:   %pattern_type.012: type = pattern_type %VirtualBase [concrete]
// CHECK:STDOUT:   %VirtualBase.F.type: type = fn_type @VirtualBase.F [concrete]
// CHECK:STDOUT:   %VirtualBase.F: %VirtualBase.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %VirtualBase.vtable_decl: ref %ptr.454 = vtable_decl @VirtualBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %VirtualIntermediate: type = class_type @VirtualIntermediate [concrete]
// CHECK:STDOUT:   %VirtualIntermediate.elem: type = unbound_element_type %VirtualIntermediate, %VirtualBase [concrete]
// CHECK:STDOUT:   %VirtualIntermediate.vtable_decl: ref %ptr.454 = vtable_decl @VirtualIntermediate.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.d36: type = struct_type {.base: %VirtualBase} [concrete]
// CHECK:STDOUT:   %complete_type.e60: <witness> = complete_type_witness %struct_type.base.d36 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %VirtualIntermediate [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.cec: type = struct_type {.base: %VirtualIntermediate} [concrete]
// CHECK:STDOUT:   %complete_type.4c0: <witness> = complete_type_witness %struct_type.base.cec [concrete]
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
// CHECK:STDOUT:   %VirtualBase.F.decl: %VirtualBase.F.type = fn_decl @VirtualBase.F [concrete = constants.%VirtualBase.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.012 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.012 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %VirtualBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%VirtualBase [concrete = constants.%VirtualBase]
// CHECK:STDOUT:     %self: %VirtualBase = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @VirtualBase.vtable [concrete = constants.%VirtualBase.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%VirtualBase
// CHECK:STDOUT:   .F = %VirtualBase.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualIntermediate {
// CHECK:STDOUT:   %VirtualBase.ref: type = name_ref VirtualBase, file.%VirtualBase.decl [concrete = constants.%VirtualBase]
// CHECK:STDOUT:   %.loc9: %VirtualIntermediate.elem = base_decl %VirtualBase.ref, element0 [concrete]
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @VirtualIntermediate.vtable [concrete = constants.%VirtualIntermediate.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.d36 [concrete = constants.%complete_type.e60]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
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
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cec [concrete = constants.%complete_type.4c0]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .VirtualIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %VirtualIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualBase.vtable {
// CHECK:STDOUT:   @VirtualBase.%VirtualBase.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @VirtualIntermediate.vtable {
// CHECK:STDOUT:   constants.%VirtualBase.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @VirtualBase.F(%self.param: %VirtualBase);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.101: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.5a1: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc9: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %v.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %v.patt: %pattern_type.7ce = at_binding_pattern v, %v.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:     %v.param: %i32 = value_param call_param1
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %v: %i32 = value_binding v, %v.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.5a1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.F(%self.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived, %v.param: %i32);
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
// CHECK:STDOUT:   %ImplicitAs.type.8c9: type = facet_type <@ImplicitAs, @ImplicitAs(%T1)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @T2.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b8b: type = pattern_type %T2 [concrete]
// CHECK:STDOUT:   %.a53: Core.Form = init_form %T1 [concrete]
// CHECK:STDOUT:   %pattern_type.818: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert.type: type = fn_type @T2.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert: %T2.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %T1.val: %T1 = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.101: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F [concrete]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %.c13: Core.Form = init_form %T2 [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.5a1: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:   impl_decl @T2.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T1)> [concrete = constants.%ImplicitAs.type.8c9]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @T2.as.ImplicitAs.impl: %T2.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert.decl: %T2.as.ImplicitAs.impl.Convert.type = fn_decl @T2.as.ImplicitAs.impl.Convert [concrete = constants.%T2.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8b = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.818 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.818 = return_slot_pattern %return.param_patt, %T1.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %.loc11: Core.Form = init_form %T1.ref [concrete = constants.%.a53]
// CHECK:STDOUT:     %self.param: %T2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @T2.as.ImplicitAs.impl.%T2.ref [concrete = constants.%T2]
// CHECK:STDOUT:     %self: %T2 = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%T2.as.ImplicitAs.impl.Convert.decl), @T2.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .Convert = %T2.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.F.decl: %Base.F.type = fn_decl @Base.F [concrete = constants.%Base.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.101 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.101 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.818 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.818 = return_slot_pattern %return.param_patt, %T1.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %.loc17: Core.Form = init_form %T1.ref [concrete = constants.%.a53]
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .F = %Base.F.decl
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc21: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b8b = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.b8b = return_slot_pattern %return.param_patt, %T2.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %.loc29: Core.Form = init_form %T2.ref [concrete = constants.%.c13]
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %T2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.5a1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @T2.as.ImplicitAs.impl.Convert(%self.param: %T2) -> out %return.param: %T1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc12_13.2: init %T1 to %return.param = class_init () [concrete = constants.%T1.val]
// CHECK:STDOUT:   %.loc12_14: init %T1 = converted %.loc12_13.1, %.loc12_13.2 [concrete = constants.%T1.val]
// CHECK:STDOUT:   return %.loc12_14 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @Base.F(%self.param: %Base) -> out %return.param: %T1;
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived) -> out %return.param: %T2;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn: <specific function> = specific_function %Base.F, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr = vtable_decl @Base.vtable [concrete]
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
// CHECK:STDOUT:     %.loc7_21.1: type = splice_block %.loc7_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T.loc7_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc7_18.1) [symbolic = %Base.F.type (constants.%Base.F.type)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type) = struct_value () [symbolic = %Base.F (constants.%Base.F)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc13_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc7_18.1) [symbolic = %Base.F.specific_fn.loc13_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc12_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc12_22.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_22.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc13_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc13_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc13_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.2f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_18.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.1e0: <witness> = require_complete_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.d5d: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.type.cb1: type = fn_type @Base.F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.eab: %Base.F.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.818: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.cd5: <specific function> = specific_function %Base.F.eab, @Base.F(%T1) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.cc2: type = struct_type {.base: %Base.d5d} [concrete]
// CHECK:STDOUT:   %complete_type.dcb: <witness> = complete_type_witness %struct_type.base.cc2 [concrete]
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
// CHECK:STDOUT:     %.loc7_21.1: type = splice_block %.loc7_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T.loc7_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc7_18.1) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc7_18.1) [symbolic = %Base.F.specific_fn.loc9_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type.0f1) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F.cd3)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type.loc8_27 (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type.loc8_27 (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Base.F.%pattern_type.loc8_43 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Base.F.%pattern_type.loc8_43 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base.d0c) = value_param call_param0
// CHECK:STDOUT:       %.loc8_29.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.d0c)] {
// CHECK:STDOUT:         %.loc8_29.2: type = specific_constant constants.%Base.d0c, @Base(constants.%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_29.2 [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base.d0c) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Base.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Base.%T.loc7_18.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @Base.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc9_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc9_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:   %.loc12: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.818 = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.818 = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:     %t.param: %T1 = value_param call_param1
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %t: %T1 = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cc2 [concrete = constants.%complete_type.dcb]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc9_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type.loc8_27: type = pattern_type %Base [symbolic = %pattern_type.loc8_27 (constants.%pattern_type.2f0)]
// CHECK:STDOUT:   %pattern_type.loc8_43: type = pattern_type %T [symbolic = %pattern_type.loc8_43 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27: <witness> = require_complete_type %Base [symbolic = %require_complete.loc8_27 (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %require_complete.loc8_43: <witness> = require_complete_type %T [symbolic = %require_complete.loc8_43 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base.d0c), %t.param: @Base.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived, %t.param: %T1) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.0f1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.cd3
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2 => constants.%Base.F.specific_fn.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.2f0
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27 => constants.%require_complete.1e0
// CHECK:STDOUT:   %require_complete.loc8_43 => constants.%require_complete.944
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_18.1 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.cb1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.eab
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2 => constants.%Base.F.specific_fn.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.d5d
// CHECK:STDOUT:   %pattern_type.loc8_27 => constants.%pattern_type.64f
// CHECK:STDOUT:   %pattern_type.loc8_43 => constants.%pattern_type.818
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc8_27 => constants.%complete_type.513
// CHECK:STDOUT:   %require_complete.loc8_43 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F [concrete]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1.G.type: type = fn_type @T1.G [concrete]
// CHECK:STDOUT:   %T1.G: %T1.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %T2.F.type: type = fn_type @T2.F [concrete]
// CHECK:STDOUT:   %T2.F: %T2.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete]
// CHECK:STDOUT:   %complete_type.ad3: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:   %T1.F.decl: %T1.F.type = fn_decl @T1.F [concrete = constants.%T1.F] {} {}
// CHECK:STDOUT:   %T1.G.decl: %T1.G.type = fn_decl @T1.G [concrete = constants.%T1.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F = %T1.F.decl
// CHECK:STDOUT:   .G = %T1.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %.loc18: %T2.elem = base_decl %T1.ref, element0 [concrete]
// CHECK:STDOUT:   %T2.F.decl: %T2.F.type = fn_decl @T2.F [concrete = constants.%T2.F] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.ad3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc18
// CHECK:STDOUT:   .F = %T2.F.decl
// CHECK:STDOUT:   extend %T1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @T1.F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @T1.G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @T2.F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_ref_self_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %pattern_type.818: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %T1.F1.type: type = fn_type @T1.F1 [concrete]
// CHECK:STDOUT:   %T1.F1: %T1.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %pattern_type.b8b: type = pattern_type %T2 [concrete]
// CHECK:STDOUT:   %T2.F1.type: type = fn_type @T2.F1 [concrete]
// CHECK:STDOUT:   %T2.F1: %T2.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T2.vtable_decl: ref %ptr.454 = vtable_decl @T2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1} [concrete]
// CHECK:STDOUT:   %complete_type.ad3: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:   %T1.F1.decl: %T1.F1.type = fn_decl @T1.F1 [concrete = constants.%T1.F1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.818 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.818 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %T1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%T1 [concrete = constants.%T1]
// CHECK:STDOUT:     %self: %T1 = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete = constants.%T1.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F1 = %T1.F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %.loc9: %T2.elem = base_decl %T1.ref, element0 [concrete]
// CHECK:STDOUT:   %T2.F1.decl: %T2.F1.type = fn_decl @T2.F1 [concrete = constants.%T2.F1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8b = ref_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8b = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %T2 = ref_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%T2 [concrete = constants.%T2]
// CHECK:STDOUT:     %self: ref %T2 = ref_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @T2.vtable [concrete = constants.%T2.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base [concrete = constants.%complete_type.ad3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F1 = %T2.F1.decl
// CHECK:STDOUT:   extend %T1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%T1.F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T2.vtable {
// CHECK:STDOUT:   @T2.%T2.F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @T1.F1(%self.param: %T1);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @T2.F1(%self.param: ref %T2);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %pattern_type.818: type = pattern_type %T1 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F [concrete]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %T1.F.decl: %T1.F.type = fn_decl @T1.F [concrete = constants.%T1.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.818 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.818 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %T1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%T1 [concrete = constants.%T1]
// CHECK:STDOUT:     %self: %T1 = value_binding self, %self.param
// CHECK:STDOUT:     %.loc9_32.1: type = splice_block %.loc9_32.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_29.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_29.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F = %T1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(%T.loc9_29.2: type) {
// CHECK:STDOUT:   %T.loc9_29.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_29.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: %T1);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T) {
// CHECK:STDOUT:   %T.loc9_29.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_generic_virtual_in_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T.67d) [symbolic]
// CHECK:STDOUT:   %pattern_type.253: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %T.091: type = symbolic_binding T, 1 [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.67d) [symbolic]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [symbolic]
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
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.loc4_16.1) [symbolic = %T1.F.type (constants.%T1.F.type)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type) = struct_value () [symbolic = %T1.F (constants.%T1.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type (%pattern_type.253) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type (%pattern_type.253) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc9_22.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc9_22.2: type = specific_constant constants.%T1, @T1(constants.%T.67d) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_22.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1) = value_binding self, %self.param
// CHECK:STDOUT:       %.loc9_32.1: type = splice_block %.loc9_32.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc9_32.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %T.loc9_29.2: type = symbolic_binding T, 1 [symbolic = %T.loc9_29.1 (constants.%T.091)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .F = %T1.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%T.loc4_16.2: type, %T.loc9_29.2: type) {
// CHECK:STDOUT:   %T.loc9_22: type = symbolic_binding T, 0 [symbolic = %T.loc9_22 (constants.%T.67d)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T.loc9_22) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.253)]
// CHECK:STDOUT:   %T.loc9_29.1: type = symbolic_binding T, 1 [symbolic = %T.loc9_29.1 (constants.%T.091)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @T1.F.%T1 (%T1));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T.67d, constants.%T.091) {
// CHECK:STDOUT:   %T.loc9_22 => constants.%T.67d
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253
// CHECK:STDOUT:   %T.loc9_29.1 => constants.%T.091
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_with_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.253: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.F.specific_fn: <specific function> = specific_function %T1.F, @T1.F(%T) [symbolic]
// CHECK:STDOUT:   %T1.vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete]
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
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.loc4_16.1) [symbolic = %T1.F.type (constants.%T1.F.type)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type) = struct_value () [symbolic = %T1.F (constants.%T1.F)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2: <specific function> = specific_function %T1.F, @T1.F(%T.loc4_16.1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type (%pattern_type.253) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type (%pattern_type.253) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T1.F.specific_fn.loc6_1.1: <specific function> = specific_function %T1.F.decl, @T1.F(constants.%T) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete = constants.%T1.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .F = %T1.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.253)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @T1.F.%T1 (%T1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type
// CHECK:STDOUT:   %T1.F => constants.%T1.F
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_dependent_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.253: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.F.specific_fn: <specific function> = specific_function %T1.F, @T1.F(%T) [symbolic]
// CHECK:STDOUT:   %T1.vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.6d2: <witness> = require_complete_type %T1 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
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
// CHECK:STDOUT:     %.loc4_19.1: type = splice_block %.loc4_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.loc4_16.1) [symbolic = %T1.F.type (constants.%T1.F.type)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type) = struct_value () [symbolic = %T1.F (constants.%T1.F)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2: <specific function> = specific_function %T1.F, @T1.F(%T.loc4_16.1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type.loc5_27 (%pattern_type.253) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type.loc5_27 (%pattern_type.253) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @T1.F.%pattern_type.loc5_43 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @T1.F.%pattern_type.loc5_43 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @T1.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @T1.%T.loc4_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @T1.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T1.F.specific_fn.loc6_1.1: <specific function> = specific_function %T1.F.decl, @T1.F(constants.%T) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete = constants.%T1.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %T1.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type.loc5_27: type = pattern_type %T1 [symbolic = %pattern_type.loc5_27 (constants.%pattern_type.253)]
// CHECK:STDOUT:   %pattern_type.loc5_43: type = pattern_type %T [symbolic = %pattern_type.loc5_43 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27: <witness> = require_complete_type %T1 [symbolic = %require_complete.loc5_27 (constants.%require_complete.6d2)]
// CHECK:STDOUT:   %require_complete.loc5_43: <witness> = require_complete_type %T [symbolic = %require_complete.loc5_43 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @T1.F.%T1 (%T1), %t.param: @T1.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type
// CHECK:STDOUT:   %T1.F => constants.%T1.F
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.253
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.6d2
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.944
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
// CHECK:STDOUT:   %Modifiers.import_ref.d80231.2 = import_ref Modifiers//default, loc6_1, unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Modifiers.import_ref.d82 = import_ref Modifiers//default, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Modifiers.import_ref.d40 = import_ref Modifiers//default, loc5_29, unloaded
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
// CHECK:STDOUT:     %b.param_patt: %pattern_type = value_param_pattern [concrete]
// CHECK:STDOUT:     %b.patt: %pattern_type = at_binding_pattern b, %b.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %b.param: %Base = value_param call_param0
// CHECK:STDOUT:     %.loc6: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:       %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:       %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %b: %Base = value_binding b, %b.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:   vtable_decl = imports.%Modifiers.import_ref.d80231.2
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.d82
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.d40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%b.param: %Base);
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_derived_from_nongeneric.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NonGenericBase: type = class_type @NonGenericBase [concrete]
// CHECK:STDOUT:   %pattern_type.126: type = pattern_type %NonGenericBase [concrete]
// CHECK:STDOUT:   %NonGenericBase.F1.type: type = fn_type @NonGenericBase.F1 [concrete]
// CHECK:STDOUT:   %NonGenericBase.F1: %NonGenericBase.F1.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonGenericBase.F2.type: type = fn_type @NonGenericBase.F2 [concrete]
// CHECK:STDOUT:   %NonGenericBase.F2: %NonGenericBase.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %NonGenericBase.vtable_decl: ref %ptr.454 = vtable_decl @NonGenericBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericDerived.type: type = generic_class_type @GenericDerived [concrete]
// CHECK:STDOUT:   %GenericDerived.generic: %GenericDerived.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.elem: type = unbound_element_type %GenericDerived, %NonGenericBase [symbolic]
// CHECK:STDOUT:   %pattern_type.945: type = pattern_type %GenericDerived [symbolic]
// CHECK:STDOUT:   %GenericDerived.F2.type: type = fn_type @GenericDerived.F2, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.F2: %GenericDerived.F2.type = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericDerived.F3.type: type = fn_type @GenericDerived.F3, @GenericDerived(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.F3: %GenericDerived.F3.type = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericDerived.F2.specific_fn: <specific function> = specific_function %GenericDerived.F2, @GenericDerived.F2(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.F3.specific_fn: <specific function> = specific_function %GenericDerived.F3, @GenericDerived.F3(%T) [symbolic]
// CHECK:STDOUT:   %GenericDerived.vtable_decl: ref %ptr.454 = vtable_decl @GenericDerived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.66e: type = struct_type {.base: %NonGenericBase} [concrete]
// CHECK:STDOUT:   %complete_type.a92: <witness> = complete_type_witness %struct_type.base.66e [concrete]
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
// CHECK:STDOUT:     %.loc9_31.1: type = splice_block %.loc9_31.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc9_31.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_28.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonGenericBase {
// CHECK:STDOUT:   %NonGenericBase.F1.decl: %NonGenericBase.F1.type = fn_decl @NonGenericBase.F1 [concrete = constants.%NonGenericBase.F1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.126 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.126 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericBase [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %self: %NonGenericBase = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonGenericBase.F2.decl: %NonGenericBase.F2.type = fn_decl @NonGenericBase.F2 [concrete = constants.%NonGenericBase.F2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.126 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.126 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericBase [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %self: %NonGenericBase = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @NonGenericBase.vtable [concrete = constants.%NonGenericBase.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonGenericBase
// CHECK:STDOUT:   .F1 = %NonGenericBase.F1.decl
// CHECK:STDOUT:   .F2 = %NonGenericBase.F2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericDerived(%T.loc9_28.2: type) {
// CHECK:STDOUT:   %T.loc9_28.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T.loc9_28.1) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %GenericDerived.elem: type = unbound_element_type %GenericDerived, constants.%NonGenericBase [symbolic = %GenericDerived.elem (constants.%GenericDerived.elem)]
// CHECK:STDOUT:   %GenericDerived.F2.type: type = fn_type @GenericDerived.F2, @GenericDerived(%T.loc9_28.1) [symbolic = %GenericDerived.F2.type (constants.%GenericDerived.F2.type)]
// CHECK:STDOUT:   %GenericDerived.F2: @GenericDerived.%GenericDerived.F2.type (%GenericDerived.F2.type) = struct_value () [symbolic = %GenericDerived.F2 (constants.%GenericDerived.F2)]
// CHECK:STDOUT:   %GenericDerived.F3.type: type = fn_type @GenericDerived.F3, @GenericDerived(%T.loc9_28.1) [symbolic = %GenericDerived.F3.type (constants.%GenericDerived.F3.type)]
// CHECK:STDOUT:   %GenericDerived.F3: @GenericDerived.%GenericDerived.F3.type (%GenericDerived.F3.type) = struct_value () [symbolic = %GenericDerived.F3 (constants.%GenericDerived.F3)]
// CHECK:STDOUT:   %GenericDerived.F2.specific_fn.loc13_1.2: <specific function> = specific_function %GenericDerived.F2, @GenericDerived.F2(%T.loc9_28.1) [symbolic = %GenericDerived.F2.specific_fn.loc13_1.2 (constants.%GenericDerived.F2.specific_fn)]
// CHECK:STDOUT:   %GenericDerived.F3.specific_fn.loc13_1.2: <specific function> = specific_function %GenericDerived.F3, @GenericDerived.F3(%T.loc9_28.1) [symbolic = %GenericDerived.F3.specific_fn.loc13_1.2 (constants.%GenericDerived.F3.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %NonGenericBase.ref: type = name_ref NonGenericBase, file.%NonGenericBase.decl [concrete = constants.%NonGenericBase]
// CHECK:STDOUT:     %.loc10: @GenericDerived.%GenericDerived.elem (%GenericDerived.elem) = base_decl %NonGenericBase.ref, element0 [concrete]
// CHECK:STDOUT:     %GenericDerived.F2.decl: @GenericDerived.%GenericDerived.F2.type (%GenericDerived.F2.type) = fn_decl @GenericDerived.F2 [symbolic = @GenericDerived.%GenericDerived.F2 (constants.%GenericDerived.F2)] {
// CHECK:STDOUT:       %self.param_patt: @GenericDerived.F2.%pattern_type (%pattern_type.945) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @GenericDerived.F2.%pattern_type (%pattern_type.945) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @GenericDerived.F2.%GenericDerived (%GenericDerived) = value_param call_param0
// CHECK:STDOUT:       %.loc11_31.1: type = splice_block %Self.ref [symbolic = %GenericDerived (constants.%GenericDerived)] {
// CHECK:STDOUT:         %.loc11_31.2: type = specific_constant constants.%GenericDerived, @GenericDerived(constants.%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc11_31.2 [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GenericDerived.F2.%GenericDerived (%GenericDerived) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GenericDerived.F3.decl: @GenericDerived.%GenericDerived.F3.type (%GenericDerived.F3.type) = fn_decl @GenericDerived.F3 [symbolic = @GenericDerived.%GenericDerived.F3 (constants.%GenericDerived.F3)] {
// CHECK:STDOUT:       %self.param_patt: @GenericDerived.F3.%pattern_type (%pattern_type.945) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @GenericDerived.F3.%pattern_type (%pattern_type.945) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @GenericDerived.F3.%GenericDerived (%GenericDerived) = value_param call_param0
// CHECK:STDOUT:       %.loc12_30.1: type = splice_block %Self.ref [symbolic = %GenericDerived (constants.%GenericDerived)] {
// CHECK:STDOUT:         %.loc12_30.2: type = specific_constant constants.%GenericDerived, @GenericDerived(constants.%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc12_30.2 [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GenericDerived.F3.%GenericDerived (%GenericDerived) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GenericDerived.F2.specific_fn.loc13_1.1: <specific function> = specific_function %GenericDerived.F2.decl, @GenericDerived.F2(constants.%T) [symbolic = %GenericDerived.F2.specific_fn.loc13_1.2 (constants.%GenericDerived.F2.specific_fn)]
// CHECK:STDOUT:     %GenericDerived.F3.specific_fn.loc13_1.1: <specific function> = specific_function %GenericDerived.F3.decl, @GenericDerived.F3(constants.%T) [symbolic = %GenericDerived.F3.specific_fn.loc13_1.2 (constants.%GenericDerived.F3.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @GenericDerived.vtable [concrete = constants.%GenericDerived.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.base.66e [concrete = constants.%complete_type.a92]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericDerived
// CHECK:STDOUT:     .NonGenericBase = <poisoned>
// CHECK:STDOUT:     .base = %.loc10
// CHECK:STDOUT:     .F2 = %GenericDerived.F2.decl
// CHECK:STDOUT:     .F3 = %GenericDerived.F3.decl
// CHECK:STDOUT:     extend %NonGenericBase.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonGenericBase.vtable {
// CHECK:STDOUT:   @NonGenericBase.%NonGenericBase.F1.decl
// CHECK:STDOUT:   @NonGenericBase.%NonGenericBase.F2.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @GenericDerived.vtable {
// CHECK:STDOUT:   constants.%NonGenericBase.F1
// CHECK:STDOUT:   @GenericDerived.%GenericDerived.F2.specific_fn.loc13_1.1
// CHECK:STDOUT:   @GenericDerived.%GenericDerived.F3.specific_fn.loc13_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @NonGenericBase.F1(%self.param: %NonGenericBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @NonGenericBase.F2(%self.param: %NonGenericBase) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic override fn @GenericDerived.F2(@GenericDerived.%T.loc9_28.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericDerived [symbolic = %pattern_type (constants.%pattern_type.945)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericDerived [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   override fn(%self.param: @GenericDerived.F2.%GenericDerived (%GenericDerived)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @GenericDerived.F3(@GenericDerived.%T.loc9_28.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericDerived: type = class_type @GenericDerived, @GenericDerived(%T) [symbolic = %GenericDerived (constants.%GenericDerived)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericDerived [symbolic = %pattern_type (constants.%pattern_type.945)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericDerived [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @GenericDerived.F3.%GenericDerived (%GenericDerived)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericDerived(constants.%T) {
// CHECK:STDOUT:   %T.loc9_28.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %GenericDerived.elem => constants.%GenericDerived.elem
// CHECK:STDOUT:   %GenericDerived.F2.type => constants.%GenericDerived.F2.type
// CHECK:STDOUT:   %GenericDerived.F2 => constants.%GenericDerived.F2
// CHECK:STDOUT:   %GenericDerived.F3.type => constants.%GenericDerived.F3.type
// CHECK:STDOUT:   %GenericDerived.F3 => constants.%GenericDerived.F3
// CHECK:STDOUT:   %GenericDerived.F2.specific_fn.loc13_1.2 => constants.%GenericDerived.F2.specific_fn
// CHECK:STDOUT:   %GenericDerived.F3.specific_fn.loc13_1.2 => constants.%GenericDerived.F3.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericDerived.F2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.945
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericDerived.F3(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericDerived => constants.%GenericDerived
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.945
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nongeneric_derived_from_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %GenericBase.type: type = generic_class_type @GenericBase [concrete]
// CHECK:STDOUT:   %GenericBase.generic: %GenericBase.type = struct_value () [concrete]
// CHECK:STDOUT:   %GenericBase.5e3: type = class_type @GenericBase, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.3f7: type = pattern_type %GenericBase.5e3 [symbolic]
// CHECK:STDOUT:   %GenericBase.F1.type.c30: type = fn_type @GenericBase.F1, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %GenericBase.F1.5f6: %GenericBase.F1.type.c30 = struct_value () [symbolic]
// CHECK:STDOUT:   %GenericBase.F2.type.3d4: type = fn_type @GenericBase.F2, @GenericBase(%T) [symbolic]
// CHECK:STDOUT:   %GenericBase.F2.9ed: %GenericBase.F2.type.3d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %GenericBase.F1.specific_fn.463: <specific function> = specific_function %GenericBase.F1.5f6, @GenericBase.F1(%T) [symbolic]
// CHECK:STDOUT:   %GenericBase.F2.specific_fn.92d: <specific function> = specific_function %GenericBase.F2.9ed, @GenericBase.F2(%T) [symbolic]
// CHECK:STDOUT:   %GenericBase.vtable_decl: ref %ptr.454 = vtable_decl @GenericBase.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase.5e3 [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %NonGenericDerived: type = class_type @NonGenericDerived [concrete]
// CHECK:STDOUT:   %GenericBase.52d: type = class_type @GenericBase, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %GenericBase.F1.type.93d: type = fn_type @GenericBase.F1, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %GenericBase.F1.c55: %GenericBase.F1.type.93d = struct_value () [concrete]
// CHECK:STDOUT:   %GenericBase.F2.type.14d: type = fn_type @GenericBase.F2, @GenericBase(%T1) [concrete]
// CHECK:STDOUT:   %GenericBase.F2.301: %GenericBase.F2.type.14d = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.889: type = pattern_type %GenericBase.52d [concrete]
// CHECK:STDOUT:   %GenericBase.F1.specific_fn.9eb: <specific function> = specific_function %GenericBase.F1.c55, @GenericBase.F1(%T1) [concrete]
// CHECK:STDOUT:   %GenericBase.F2.specific_fn.ed3: <specific function> = specific_function %GenericBase.F2.301, @GenericBase.F2(%T1) [concrete]
// CHECK:STDOUT:   %NonGenericDerived.elem: type = unbound_element_type %NonGenericDerived, %GenericBase.52d [concrete]
// CHECK:STDOUT:   %pattern_type.c19: type = pattern_type %NonGenericDerived [concrete]
// CHECK:STDOUT:   %NonGenericDerived.F2.type: type = fn_type @NonGenericDerived.F2 [concrete]
// CHECK:STDOUT:   %NonGenericDerived.F2: %NonGenericDerived.F2.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonGenericDerived.F3.type: type = fn_type @NonGenericDerived.F3 [concrete]
// CHECK:STDOUT:   %NonGenericDerived.F3: %NonGenericDerived.F3.type = struct_value () [concrete]
// CHECK:STDOUT:   %NonGenericDerived.vtable_decl: ref %ptr.454 = vtable_decl @NonGenericDerived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.19b: type = struct_type {.base: %GenericBase.52d} [concrete]
// CHECK:STDOUT:   %complete_type.1f1: <witness> = complete_type_witness %struct_type.base.19b [concrete]
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
// CHECK:STDOUT:     %.loc4_28.1: type = splice_block %.loc4_28.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_28.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_25.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_25.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %NonGenericDerived.decl: type = class_decl @NonGenericDerived [concrete = constants.%NonGenericDerived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @GenericBase(%T.loc4_25.2: type) {
// CHECK:STDOUT:   %T.loc4_25.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_25.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericBase.F1.type: type = fn_type @GenericBase.F1, @GenericBase(%T.loc4_25.1) [symbolic = %GenericBase.F1.type (constants.%GenericBase.F1.type.c30)]
// CHECK:STDOUT:   %GenericBase.F1: @GenericBase.%GenericBase.F1.type (%GenericBase.F1.type.c30) = struct_value () [symbolic = %GenericBase.F1 (constants.%GenericBase.F1.5f6)]
// CHECK:STDOUT:   %GenericBase.F2.type: type = fn_type @GenericBase.F2, @GenericBase(%T.loc4_25.1) [symbolic = %GenericBase.F2.type (constants.%GenericBase.F2.type.3d4)]
// CHECK:STDOUT:   %GenericBase.F2: @GenericBase.%GenericBase.F2.type (%GenericBase.F2.type.3d4) = struct_value () [symbolic = %GenericBase.F2 (constants.%GenericBase.F2.9ed)]
// CHECK:STDOUT:   %GenericBase.F1.specific_fn.loc7_1.2: <specific function> = specific_function %GenericBase.F1, @GenericBase.F1(%T.loc4_25.1) [symbolic = %GenericBase.F1.specific_fn.loc7_1.2 (constants.%GenericBase.F1.specific_fn.463)]
// CHECK:STDOUT:   %GenericBase.F2.specific_fn.loc7_1.2: <specific function> = specific_function %GenericBase.F2, @GenericBase.F2(%T.loc4_25.1) [symbolic = %GenericBase.F2.specific_fn.loc7_1.2 (constants.%GenericBase.F2.specific_fn.92d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %GenericBase.F1.decl: @GenericBase.%GenericBase.F1.type (%GenericBase.F1.type.c30) = fn_decl @GenericBase.F1 [symbolic = @GenericBase.%GenericBase.F1 (constants.%GenericBase.F1.5f6)] {
// CHECK:STDOUT:       %self.param_patt: @GenericBase.F1.%pattern_type (%pattern_type.3f7) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @GenericBase.F1.%pattern_type (%pattern_type.3f7) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @GenericBase.F1.%GenericBase (%GenericBase.5e3) = value_param call_param0
// CHECK:STDOUT:       %.loc5_30.1: type = splice_block %Self.ref [symbolic = %GenericBase (constants.%GenericBase.5e3)] {
// CHECK:STDOUT:         %.loc5_30.2: type = specific_constant constants.%GenericBase.5e3, @GenericBase(constants.%T) [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_30.2 [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GenericBase.F1.%GenericBase (%GenericBase.5e3) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GenericBase.F2.decl: @GenericBase.%GenericBase.F2.type (%GenericBase.F2.type.3d4) = fn_decl @GenericBase.F2 [symbolic = @GenericBase.%GenericBase.F2 (constants.%GenericBase.F2.9ed)] {
// CHECK:STDOUT:       %self.param_patt: @GenericBase.F2.%pattern_type (%pattern_type.3f7) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @GenericBase.F2.%pattern_type (%pattern_type.3f7) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @GenericBase.F2.%GenericBase (%GenericBase.5e3) = value_param call_param0
// CHECK:STDOUT:       %.loc6_30.1: type = splice_block %Self.ref [symbolic = %GenericBase (constants.%GenericBase.5e3)] {
// CHECK:STDOUT:         %.loc6_30.2: type = specific_constant constants.%GenericBase.5e3, @GenericBase(constants.%T) [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_30.2 [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @GenericBase.F2.%GenericBase (%GenericBase.5e3) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %GenericBase.F1.specific_fn.loc7_1.1: <specific function> = specific_function %GenericBase.F1.decl, @GenericBase.F1(constants.%T) [symbolic = %GenericBase.F1.specific_fn.loc7_1.2 (constants.%GenericBase.F1.specific_fn.463)]
// CHECK:STDOUT:     %GenericBase.F2.specific_fn.loc7_1.1: <specific function> = specific_function %GenericBase.F2.decl, @GenericBase.F2(constants.%T) [symbolic = %GenericBase.F2.specific_fn.loc7_1.2 (constants.%GenericBase.F2.specific_fn.92d)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @GenericBase.vtable [concrete = constants.%GenericBase.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%GenericBase.5e3
// CHECK:STDOUT:     .F1 = %GenericBase.F1.decl
// CHECK:STDOUT:     .F2 = %GenericBase.F2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @NonGenericDerived {
// CHECK:STDOUT:   %GenericBase.ref: %GenericBase.type = name_ref GenericBase, file.%GenericBase.decl [concrete = constants.%GenericBase.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(constants.%T1) [concrete = constants.%GenericBase.52d]
// CHECK:STDOUT:   %.loc12: %NonGenericDerived.elem = base_decl %GenericBase, element0 [concrete]
// CHECK:STDOUT:   %NonGenericDerived.F2.decl: %NonGenericDerived.F2.type = fn_decl @NonGenericDerived.F2 [concrete = constants.%NonGenericDerived.F2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c19 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.c19 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericDerived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericDerived [concrete = constants.%NonGenericDerived]
// CHECK:STDOUT:     %self: %NonGenericDerived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %NonGenericDerived.F3.decl: %NonGenericDerived.F3.type = fn_decl @NonGenericDerived.F3 [concrete = constants.%NonGenericDerived.F3] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c19 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.c19 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %NonGenericDerived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%NonGenericDerived [concrete = constants.%NonGenericDerived]
// CHECK:STDOUT:     %self: %NonGenericDerived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @NonGenericDerived.vtable [concrete = constants.%NonGenericDerived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.19b [concrete = constants.%complete_type.1f1]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NonGenericDerived
// CHECK:STDOUT:   .GenericBase = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .F2 = %NonGenericDerived.F2.decl
// CHECK:STDOUT:   .F3 = %NonGenericDerived.F3.decl
// CHECK:STDOUT:   extend %GenericBase
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @GenericBase.vtable {
// CHECK:STDOUT:   @GenericBase.%GenericBase.F1.specific_fn.loc7_1.1
// CHECK:STDOUT:   @GenericBase.%GenericBase.F2.specific_fn.loc7_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @NonGenericDerived.vtable {
// CHECK:STDOUT:   constants.%GenericBase.F1.specific_fn.9eb
// CHECK:STDOUT:   @NonGenericDerived.%NonGenericDerived.F2.decl
// CHECK:STDOUT:   @NonGenericDerived.%NonGenericDerived.F3.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @GenericBase.F1(@GenericBase.%T.loc4_25.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(%T) [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericBase [symbolic = %pattern_type (constants.%pattern_type.3f7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @GenericBase.F1.%GenericBase (%GenericBase.5e3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @GenericBase.F2(@GenericBase.%T.loc4_25.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %GenericBase: type = class_type @GenericBase, @GenericBase(%T) [symbolic = %GenericBase (constants.%GenericBase.5e3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %GenericBase [symbolic = %pattern_type (constants.%pattern_type.3f7)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %GenericBase [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @GenericBase.F2.%GenericBase (%GenericBase.5e3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @NonGenericDerived.F2(%self.param: %NonGenericDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @NonGenericDerived.F3(%self.param: %NonGenericDerived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase(constants.%T) {
// CHECK:STDOUT:   %T.loc4_25.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericBase.F1.type => constants.%GenericBase.F1.type.c30
// CHECK:STDOUT:   %GenericBase.F1 => constants.%GenericBase.F1.5f6
// CHECK:STDOUT:   %GenericBase.F2.type => constants.%GenericBase.F2.type.3d4
// CHECK:STDOUT:   %GenericBase.F2 => constants.%GenericBase.F2.9ed
// CHECK:STDOUT:   %GenericBase.F1.specific_fn.loc7_1.2 => constants.%GenericBase.F1.specific_fn.463
// CHECK:STDOUT:   %GenericBase.F2.specific_fn.loc7_1.2 => constants.%GenericBase.F2.specific_fn.92d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase.F1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.5e3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase.F2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.5e3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3f7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase(constants.%T1) {
// CHECK:STDOUT:   %T.loc4_25.1 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %GenericBase.F1.type => constants.%GenericBase.F1.type.93d
// CHECK:STDOUT:   %GenericBase.F1 => constants.%GenericBase.F1.c55
// CHECK:STDOUT:   %GenericBase.F2.type => constants.%GenericBase.F2.type.14d
// CHECK:STDOUT:   %GenericBase.F2 => constants.%GenericBase.F2.301
// CHECK:STDOUT:   %GenericBase.F1.specific_fn.loc7_1.2 => constants.%GenericBase.F1.specific_fn.9eb
// CHECK:STDOUT:   %GenericBase.F2.specific_fn.loc7_1.2 => constants.%GenericBase.F2.specific_fn.ed3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase.F1(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.52d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.889
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @GenericBase.F2(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %GenericBase => constants.%GenericBase.52d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.889
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_specifically.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %ptr.4b1: type = ptr_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %pattern_type.cf5: type = pattern_type %ptr.4b1 [symbolic]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.1e0: <witness> = require_complete_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %require_complete.3cf: <witness> = require_complete_type %ptr.4b1 [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %D1: type = class_type @D1 [concrete]
// CHECK:STDOUT:   %Base.d5d: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.type.cb1: type = fn_type @Base.F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.eab: %Base.F.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %Base.d5d [concrete]
// CHECK:STDOUT:   %ptr.6e0: type = ptr_type %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.912: type = pattern_type %ptr.6e0 [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.cd5: <specific function> = specific_function %Base.F.eab, @Base.F(%T1) [concrete]
// CHECK:STDOUT:   %D1.elem: type = unbound_element_type %D1, %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.7f9: type = pattern_type %D1 [concrete]
// CHECK:STDOUT:   %D1.F.type: type = fn_type @D1.F [concrete]
// CHECK:STDOUT:   %D1.F: %D1.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D1.vtable_decl: ref %ptr.454 = vtable_decl @D1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.cc2: type = struct_type {.base: %Base.d5d} [concrete]
// CHECK:STDOUT:   %complete_type.dcb: <witness> = complete_type_witness %struct_type.base.cc2 [concrete]
// CHECK:STDOUT:   %complete_type.96e: <witness> = complete_type_witness %ptr.6e0 [concrete]
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
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %D1.decl: type = class_decl @D1 [concrete = constants.%D1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc4_18.1) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc4_18.1) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type.0f1) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F.cd3)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.cf5) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.cf5) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base.loc5_29 (%Base.d0c) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %Base.loc5_29 (constants.%Base.d0c)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%Base.d0c, @Base(constants.%T) [symbolic = %Base.loc5_29 (constants.%Base.d0c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %Base.loc5_29 (constants.%Base.d0c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base.loc5_29 (%Base.d0c) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Base.F.%ptr.loc5_52.1 (%ptr.4b1) = value_param call_param1
// CHECK:STDOUT:       %.loc5_52: type = splice_block %ptr.loc5_52.2 [symbolic = %ptr.loc5_52.1 (constants.%ptr.4b1)] {
// CHECK:STDOUT:         %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Base.%T.loc4_18.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %Base.loc5_51: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc5_29 (constants.%Base.d0c)]
// CHECK:STDOUT:         %ptr.loc5_52.2: type = ptr_type %Base.loc5_51 [symbolic = %ptr.loc5_52.1 (constants.%ptr.4b1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @Base.F.%ptr.loc5_52.1 (%ptr.4b1) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc6_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:   %.loc9: %D1.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %D1.F.decl: %D1.F.type = fn_decl @D1.F [concrete = constants.%D1.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7f9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7f9 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.912 = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.912 = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D1 [concrete = constants.%D1]
// CHECK:STDOUT:     %self: %D1 = value_binding self, %self.param
// CHECK:STDOUT:     %t.param: %ptr.6e0 = value_param call_param1
// CHECK:STDOUT:     %.loc10: type = splice_block %ptr [concrete = constants.%ptr.6e0] {
// CHECK:STDOUT:       %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:       %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:       %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:       %ptr: type = ptr_type %Base [concrete = constants.%ptr.6e0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %ptr.6e0 = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @D1.vtable [concrete = constants.%D1.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cc2 [concrete = constants.%complete_type.dcb]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %D1.F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @D1.vtable {
// CHECK:STDOUT:   @D1.%D1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base.loc5_29: type = class_type @Base, @Base(%T) [symbolic = %Base.loc5_29 (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type.loc5_27: type = pattern_type %Base.loc5_29 [symbolic = %pattern_type.loc5_27 (constants.%pattern_type.2f0)]
// CHECK:STDOUT:   %ptr.loc5_52.1: type = ptr_type %Base.loc5_29 [symbolic = %ptr.loc5_52.1 (constants.%ptr.4b1)]
// CHECK:STDOUT:   %pattern_type.loc5_43: type = pattern_type %ptr.loc5_52.1 [symbolic = %pattern_type.loc5_43 (constants.%pattern_type.cf5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27: <witness> = require_complete_type %Base.loc5_29 [symbolic = %require_complete.loc5_27 (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %require_complete.loc5_43: <witness> = require_complete_type %ptr.loc5_52.1 [symbolic = %require_complete.loc5_43 (constants.%require_complete.3cf)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base.loc5_29 (%Base.d0c), %t.param: @Base.F.%ptr.loc5_52.1 (%ptr.4b1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @D1.F(%self.param: %D1, %t.param: %ptr.6e0) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.0f1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.cd3
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base.loc5_29 => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.2f0
// CHECK:STDOUT:   %ptr.loc5_52.1 => constants.%ptr.4b1
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.cf5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.1e0
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.3cf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.cb1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.eab
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base.loc5_29 => constants.%Base.d5d
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.64f
// CHECK:STDOUT:   %ptr.loc5_52.1 => constants.%ptr.6e0
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.912
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%complete_type.513
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%complete_type.96e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_generic_specifically_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %ptr.9ce: type = ptr_type %T1 [concrete]
// CHECK:STDOUT:   %pattern_type.c16: type = pattern_type %ptr.9ce [concrete]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %D1: type = class_type @D1 [concrete]
// CHECK:STDOUT:   %Base.d5d: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.type.cb1: type = fn_type @Base.F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.eab: %Base.F.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %Base.d5d [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.cd5: <specific function> = specific_function %Base.F.eab, @Base.F(%T1) [concrete]
// CHECK:STDOUT:   %D1.elem: type = unbound_element_type %D1, %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.7f9: type = pattern_type %D1 [concrete]
// CHECK:STDOUT:   %ptr.9f6: type = ptr_type %T2 [concrete]
// CHECK:STDOUT:   %pattern_type.64c: type = pattern_type %ptr.9f6 [concrete]
// CHECK:STDOUT:   %D1.F.type: type = fn_type @D1.F [concrete]
// CHECK:STDOUT:   %D1.F: %D1.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %D1.vtable_decl: ref %ptr.454 = vtable_decl @D1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.cc2: type = struct_type {.base: %Base.d5d} [concrete]
// CHECK:STDOUT:   %complete_type.dcb: <witness> = complete_type_witness %struct_type.base.cc2 [concrete]
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
// CHECK:STDOUT:     %.loc7_21.1: type = splice_block %.loc7_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D1.decl: type = class_decl @D1 [concrete = constants.%D1] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T.loc7_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc7_18.1) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc7_18.1) [symbolic = %Base.F.specific_fn.loc9_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type.0f1) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F.cd3)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: %pattern_type.c16 = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: %pattern_type.c16 = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base.d0c) = value_param call_param0
// CHECK:STDOUT:       %.loc8_29.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.d0c)] {
// CHECK:STDOUT:         %.loc8_29.2: type = specific_constant constants.%Base.d0c, @Base(constants.%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_29.2 [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base.d0c) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: %ptr.9ce = value_param call_param1
// CHECK:STDOUT:       %.loc8_47: type = splice_block %ptr [concrete = constants.%ptr.9ce] {
// CHECK:STDOUT:         %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:         %ptr: type = ptr_type %T1.ref [concrete = constants.%ptr.9ce]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: %ptr.9ce = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc9_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc9_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:     .T2 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:   %.loc12: %D1.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %D1.F.decl: %D1.F.type = fn_decl @D1.F [concrete = constants.%D1.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.7f9 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.7f9 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.64c = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.64c = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D1 [concrete = constants.%D1]
// CHECK:STDOUT:     %self: %D1 = value_binding self, %self.param
// CHECK:STDOUT:     %t.param: %ptr.9f6 = value_param call_param1
// CHECK:STDOUT:     %.loc20: type = splice_block %ptr [concrete = constants.%ptr.9f6] {
// CHECK:STDOUT:       %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:       %ptr: type = ptr_type %T2.ref [concrete = constants.%ptr.9f6]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %ptr.9f6 = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @D1.vtable [concrete = constants.%D1.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cc2 [concrete = constants.%complete_type.dcb]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc12
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT:   .F = %D1.F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc9_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @D1.vtable {
// CHECK:STDOUT:   @D1.%D1.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc7_18.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.2f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base.d0c), %t.param: %ptr.9ce) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @D1.F(%self.param: %D1, %t.param: %ptr.9f6) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.0f1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.cd3
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2 => constants.%Base.F.specific_fn.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_18.1 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.cb1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.eab
// CHECK:STDOUT:   %Base.F.specific_fn.loc9_1.2 => constants.%Base.F.specific_fn.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.d5d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.64f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_generic_generic_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn: <specific function> = specific_function %Base.F, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.1e0: <witness> = require_complete_type %Base [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [symbolic]
// CHECK:STDOUT:   %pattern_type.b9d: type = pattern_type %Derived [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Derived.F.specific_fn: <specific function> = specific_function %Derived.F, @Derived.F(%T) [symbolic]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.b5f: type = struct_type {.base: %Base} [symbolic]
// CHECK:STDOUT:   %complete_type.33f: <witness> = complete_type_witness %struct_type.base.b5f [symbolic]
// CHECK:STDOUT:   %require_complete.a43: <witness> = require_complete_type %Derived [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
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
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc4_22.1) [symbolic = %Base.F.type (constants.%Base.F.type)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type) = struct_value () [symbolic = %Base.F (constants.%Base.F)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc4_22.1) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.4f4) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Base.F.%ptr.loc5_46.1 (%ptr.e8f) = value_param call_param1
// CHECK:STDOUT:       %.loc5_46: type = splice_block %ptr.loc5_46.2 [symbolic = %ptr.loc5_46.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Base.%T.loc4_22.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %ptr.loc5_46.2: type = ptr_type %T.ref [symbolic = %ptr.loc5_46.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @Base.F.%ptr.loc5_46.1 (%ptr.e8f) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc6_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc7_16.2: type) {
// CHECK:STDOUT:   %T.loc7_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc8_22.2: type = class_type @Base, @Base(%T.loc7_16.1) [symbolic = %Base.loc8_22.2 (constants.%Base)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.loc8_22.2 [symbolic = %require_complete (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc7_16.1) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.loc8_22.2 [symbolic = %Derived.elem (constants.%Derived.elem)]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F, @Derived(%T.loc7_16.1) [symbolic = %Derived.F.type (constants.%Derived.F.type)]
// CHECK:STDOUT:   %Derived.F: @Derived.%Derived.F.type (%Derived.F.type) = struct_value () [symbolic = %Derived.F (constants.%Derived.F)]
// CHECK:STDOUT:   %Derived.F.specific_fn.loc17_1.2: <specific function> = specific_function %Derived.F, @Derived.F(%T.loc7_16.1) [symbolic = %Derived.F.specific_fn.loc17_1.2 (constants.%Derived.F.specific_fn)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @Derived.%Base.loc8_22.2 (%Base)} [symbolic = %struct_type.base (constants.%struct_type.base.b5f)]
// CHECK:STDOUT:   %complete_type.loc17_1.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.33f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_16.2 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:     %Base.loc8_22.1: type = class_type @Base, @Base(constants.%T) [symbolic = %Base.loc8_22.2 (constants.%Base)]
// CHECK:STDOUT:     %.loc8: @Derived.%Derived.elem (%Derived.elem) = base_decl %Base.loc8_22.1, element0 [concrete]
// CHECK:STDOUT:     %Derived.F.decl: @Derived.%Derived.F.type (%Derived.F.type) = fn_decl @Derived.F [symbolic = @Derived.%Derived.F (constants.%Derived.F)] {
// CHECK:STDOUT:       %self.param_patt: @Derived.F.%pattern_type.loc16_28 (%pattern_type.b9d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Derived.F.%pattern_type.loc16_28 (%pattern_type.b9d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Derived.F.%pattern_type.loc16_44 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Derived.F.%pattern_type.loc16_44 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Derived.F.%Derived (%Derived) = value_param call_param0
// CHECK:STDOUT:       %.loc16_30.1: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived)] {
// CHECK:STDOUT:         %.loc16_30.2: type = specific_constant constants.%Derived, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc16_30.2 [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Derived.F.%Derived (%Derived) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Derived.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Derived.%T.loc7_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @Derived.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Derived.F.specific_fn.loc17_1.1: <specific function> = specific_function %Derived.F.decl, @Derived.F(constants.%T) [symbolic = %Derived.F.specific_fn.loc17_1.2 (constants.%Derived.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:     %complete_type.loc17_1.1: <witness> = complete_type_witness constants.%struct_type.base.b5f [symbolic = %complete_type.loc17_1.2 (constants.%complete_type.33f)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc17_1.1
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     .F = %Derived.F.decl
// CHECK:STDOUT:     extend %Base.loc8_22.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.specific_fn.loc17_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type.loc5_27: type = pattern_type %Base [symbolic = %pattern_type.loc5_27 (constants.%pattern_type.2f0)]
// CHECK:STDOUT:   %ptr.loc5_46.1: type = ptr_type %T [symbolic = %ptr.loc5_46.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type.loc5_43: type = pattern_type %ptr.loc5_46.1 [symbolic = %pattern_type.loc5_43 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27: <witness> = require_complete_type %Base [symbolic = %require_complete.loc5_27 (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %require_complete.loc5_43: <witness> = require_complete_type %ptr.loc5_46.1 [symbolic = %require_complete.loc5_43 (constants.%require_complete.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base), %t.param: @Base.F.%ptr.loc5_46.1 (%ptr.e8f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic override fn @Derived.F(@Derived.%T.loc7_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %pattern_type.loc16_28: type = pattern_type %Derived [symbolic = %pattern_type.loc16_28 (constants.%pattern_type.b9d)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %require_complete.loc16_46: <witness> = require_complete_type %Base [symbolic = %require_complete.loc16_46 (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %pattern_type.loc16_44: type = pattern_type %T [symbolic = %pattern_type.loc16_44 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_28: <witness> = require_complete_type %Derived [symbolic = %require_complete.loc16_28 (constants.%require_complete.a43)]
// CHECK:STDOUT:   %require_complete.loc16_44: <witness> = require_complete_type %T [symbolic = %require_complete.loc16_44 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   override fn(%self.param: @Derived.F.%Derived (%Derived), %t.param: @Derived.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type
// CHECK:STDOUT:   %Base.F => constants.%Base.F
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.2f0
// CHECK:STDOUT:   %ptr.loc5_46.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.1e0
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc7_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.loc8_22.2 => constants.%Base
// CHECK:STDOUT:   %require_complete => constants.%require_complete.1e0
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.elem => constants.%Derived.elem
// CHECK:STDOUT:   %Derived.F.type => constants.%Derived.F.type
// CHECK:STDOUT:   %Derived.F => constants.%Derived.F
// CHECK:STDOUT:   %Derived.F.specific_fn.loc17_1.2 => constants.%Derived.F.specific_fn
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.b5f
// CHECK:STDOUT:   %complete_type.loc17_1.2 => constants.%complete_type.33f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %pattern_type.loc16_28 => constants.%pattern_type.b9d
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %require_complete.loc16_46 => constants.%require_complete.1e0
// CHECK:STDOUT:   %pattern_type.loc16_44 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc16_28 => constants.%require_complete.a43
// CHECK:STDOUT:   %require_complete.loc16_44 => constants.%require_complete.944
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.1e0: <witness> = require_complete_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Derived.type: type = generic_class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.generic: %Derived.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Base.f29: type = class_type @Base, @Base(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %Base.F.type.c53: type = fn_type @Base.F, @Base(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %Base.F.c0a: %Base.F.type.c53 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.d96: type = pattern_type %Base.f29 [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Base.F.specific_fn.35d: <specific function> = specific_function %Base.F.c0a, @Base.F(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %require_complete.b3c: <witness> = require_complete_type %Base.f29 [symbolic]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.f29 [symbolic]
// CHECK:STDOUT:   %pattern_type.b9d: type = pattern_type %Derived [symbolic]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %Derived.F.specific_fn: <specific function> = specific_function %Derived.F, @Derived.F(%T) [symbolic]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.aa5: type = struct_type {.base: %Base.f29} [symbolic]
// CHECK:STDOUT:   %complete_type.8c0: <witness> = complete_type_witness %struct_type.base.aa5 [symbolic]
// CHECK:STDOUT:   %require_complete.a43: <witness> = require_complete_type %Derived [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
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
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: %Derived.type = class_decl @Derived [concrete = constants.%Derived.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_19.1: type = splice_block %.loc7_19.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc4_22.1) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc4_22.1) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type.0f1) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F.cd3)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type.loc5_27 (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Base.F.%pattern_type.loc5_43 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base.d0c) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.d0c)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%Base.d0c, @Base(constants.%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base.d0c) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Base.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Base.%T.loc4_22.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @Base.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc6_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc7_16.2: type) {
// CHECK:STDOUT:   %T.loc7_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc8_22.2: type = ptr_type %T.loc7_16.1 [symbolic = %ptr.loc8_22.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %Base.loc8_23.2: type = class_type @Base, @Base(%ptr.loc8_22.2) [symbolic = %Base.loc8_23.2 (constants.%Base.f29)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.loc8_23.2 [symbolic = %require_complete (constants.%require_complete.b3c)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T.loc7_16.1) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.loc8_23.2 [symbolic = %Derived.elem (constants.%Derived.elem)]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F, @Derived(%T.loc7_16.1) [symbolic = %Derived.F.type (constants.%Derived.F.type)]
// CHECK:STDOUT:   %Derived.F: @Derived.%Derived.F.type (%Derived.F.type) = struct_value () [symbolic = %Derived.F (constants.%Derived.F)]
// CHECK:STDOUT:   %Derived.F.specific_fn.loc10_1.2: <specific function> = specific_function %Derived.F, @Derived.F(%T.loc7_16.1) [symbolic = %Derived.F.specific_fn.loc10_1.2 (constants.%Derived.F.specific_fn)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @Derived.%Base.loc8_23.2 (%Base.f29)} [symbolic = %struct_type.base (constants.%struct_type.base.aa5)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.8c0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_16.2 [symbolic = %T.loc7_16.1 (constants.%T)]
// CHECK:STDOUT:     %ptr.loc8_22.1: type = ptr_type %T.ref [symbolic = %ptr.loc8_22.2 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %Base.loc8_23.1: type = class_type @Base, @Base(constants.%ptr.e8f) [symbolic = %Base.loc8_23.2 (constants.%Base.f29)]
// CHECK:STDOUT:     %.loc8: @Derived.%Derived.elem (%Derived.elem) = base_decl %Base.loc8_23.1, element0 [concrete]
// CHECK:STDOUT:     %Derived.F.decl: @Derived.%Derived.F.type (%Derived.F.type) = fn_decl @Derived.F [symbolic = @Derived.%Derived.F (constants.%Derived.F)] {
// CHECK:STDOUT:       %self.param_patt: @Derived.F.%pattern_type.loc9_28 (%pattern_type.b9d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Derived.F.%pattern_type.loc9_28 (%pattern_type.b9d) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Derived.F.%pattern_type.loc9_44 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Derived.F.%pattern_type.loc9_44 (%pattern_type.4f4) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Derived.F.%Derived (%Derived) = value_param call_param0
// CHECK:STDOUT:       %.loc9_30.1: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived)] {
// CHECK:STDOUT:         %.loc9_30.2: type = specific_constant constants.%Derived, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc9_30.2 [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Derived.F.%Derived (%Derived) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Derived.F.%ptr.loc9_46 (%ptr.e8f) = value_param call_param1
// CHECK:STDOUT:       %.loc9_47: type = splice_block %ptr.loc9_47 [symbolic = %ptr.loc9_46 (constants.%ptr.e8f)] {
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Derived.%T.loc7_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %ptr.loc9_47: type = ptr_type %T.ref [symbolic = %ptr.loc9_46 (constants.%ptr.e8f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @Derived.F.%ptr.loc9_46 (%ptr.e8f) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Derived.F.specific_fn.loc10_1.1: <specific function> = specific_function %Derived.F.decl, @Derived.F(constants.%T) [symbolic = %Derived.F.specific_fn.loc10_1.2 (constants.%Derived.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.aa5 [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.8c0)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     .F = %Derived.F.decl
// CHECK:STDOUT:     extend %Base.loc8_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.specific_fn.loc10_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type.loc5_27: type = pattern_type %Base [symbolic = %pattern_type.loc5_27 (constants.%pattern_type.2f0)]
// CHECK:STDOUT:   %pattern_type.loc5_43: type = pattern_type %T [symbolic = %pattern_type.loc5_43 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27: <witness> = require_complete_type %Base [symbolic = %require_complete.loc5_27 (constants.%require_complete.1e0)]
// CHECK:STDOUT:   %require_complete.loc5_43: <witness> = require_complete_type %T [symbolic = %require_complete.loc5_43 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base.d0c), %t.param: @Base.F.%T (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic override fn @Derived.F(@Derived.%T.loc7_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %pattern_type.loc9_28: type = pattern_type %Derived [symbolic = %pattern_type.loc9_28 (constants.%pattern_type.b9d)]
// CHECK:STDOUT:   %ptr.loc9_46: type = ptr_type %T [symbolic = %ptr.loc9_46 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%ptr.loc9_46) [symbolic = %Base (constants.%Base.f29)]
// CHECK:STDOUT:   %require_complete.loc9_46: <witness> = require_complete_type %Base [symbolic = %require_complete.loc9_46 (constants.%require_complete.b3c)]
// CHECK:STDOUT:   %pattern_type.loc9_44: type = pattern_type %ptr.loc9_46 [symbolic = %pattern_type.loc9_44 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_28: <witness> = require_complete_type %Derived [symbolic = %require_complete.loc9_28 (constants.%require_complete.a43)]
// CHECK:STDOUT:   %require_complete.loc9_44: <witness> = require_complete_type %ptr.loc9_46 [symbolic = %require_complete.loc9_44 (constants.%require_complete.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   override fn(%self.param: @Derived.F.%Derived (%Derived), %t.param: @Derived.F.%ptr.loc9_46 (%ptr.e8f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.0f1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.cd3
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.2f0
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.1e0
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.944
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc7_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc8_22.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %Base.loc8_23.2 => constants.%Base.f29
// CHECK:STDOUT:   %require_complete => constants.%require_complete.b3c
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.elem => constants.%Derived.elem
// CHECK:STDOUT:   %Derived.F.type => constants.%Derived.F.type
// CHECK:STDOUT:   %Derived.F => constants.%Derived.F
// CHECK:STDOUT:   %Derived.F.specific_fn.loc10_1.2 => constants.%Derived.F.specific_fn
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.aa5
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.8c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%ptr.e8f) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%ptr.e8f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.c53
// CHECK:STDOUT:   %Base.F => constants.%Base.F.c0a
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn.35d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%ptr.e8f) {
// CHECK:STDOUT:   %T => constants.%ptr.e8f
// CHECK:STDOUT:   %Base => constants.%Base.f29
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.d96
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.b3c
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %pattern_type.loc9_28 => constants.%pattern_type.b9d
// CHECK:STDOUT:   %ptr.loc9_46 => constants.%ptr.e8f
// CHECK:STDOUT:   %Base => constants.%Base.f29
// CHECK:STDOUT:   %require_complete.loc9_46 => constants.%require_complete.b3c
// CHECK:STDOUT:   %pattern_type.loc9_44 => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9_28 => constants.%require_complete.a43
// CHECK:STDOUT:   %require_complete.loc9_44 => constants.%require_complete.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_generic_undefined.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.d5d: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.type.cb1: type = fn_type @Base.F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.eab: %Base.F.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %Base.d5d [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.cd5: <specific function> = specific_function %Base.F.eab, @Base.F(%T1) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.d5d [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F [concrete]
// CHECK:STDOUT:   %Derived.F: %Derived.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.cc2: type = struct_type {.base: %Base.d5d} [concrete]
// CHECK:STDOUT:   %complete_type.dcb: <witness> = complete_type_witness %struct_type.base.cc2 [concrete]
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
// CHECK:STDOUT:     %.loc4_25.1: type = splice_block %.loc4_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_22.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T1.decl: type = class_decl @T1 [concrete = constants.%T1] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T.loc4_22.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_22.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc4_22.1) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc4_22.1) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type.0f1) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F.cd3)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base.d0c) = value_param call_param0
// CHECK:STDOUT:       %.loc5_23.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.d0c)] {
// CHECK:STDOUT:         %.loc5_23.2: type = specific_constant constants.%Base.d0c, @Base(constants.%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_23.2 [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base.d0c) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc6_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:   %.loc11: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %Derived.F.decl: %Derived.F.type = fn_decl @Derived.F [concrete = constants.%Derived.F] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.9f6 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.cc2 [concrete = constants.%complete_type.dcb]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc11
// CHECK:STDOUT:   .F = %Derived.F.decl
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic abstract fn @Base.F(@Base.%T.loc4_22.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.2f0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   abstract fn(%self.param: @Base.F.%Base (%Base.d0c));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F(%self.param: %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc4_22.1 => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.cb1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.eab
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.d5d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.64f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_lib.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base [symbolic]
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F: %Base.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn: <specific function> = specific_function %Base.F, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
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
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_18.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T.loc4_18.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_18.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T.loc4_18.1) [symbolic = %Base.F.type (constants.%Base.F.type)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type) = struct_value () [symbolic = %Base.F (constants.%Base.F)]
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2: <specific function> = specific_function %Base.F, @Base.F(%T.loc4_18.1) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Base.F.decl: @Base.%Base.F.type (%Base.F.type) = fn_decl @Base.F [symbolic = @Base.%Base.F (constants.%Base.F)] {
// CHECK:STDOUT:       %self.param_patt: @Base.F.%pattern_type (%pattern_type.2f0) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Base.F.%pattern_type (%pattern_type.2f0) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Base.F.%Base (%Base) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%Base, @Base(constants.%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Base.F.%Base (%Base) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Base.F.specific_fn.loc6_1.1: <specific function> = specific_function %Base.F.decl, @Base.F(constants.%T) [symbolic = %Base.F.specific_fn.loc6_1.2 (constants.%Base.F.specific_fn)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base
// CHECK:STDOUT:     .F = %Base.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(@Base.%T.loc4_18.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.2f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @Base.F.%Base (%Base)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_18.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type
// CHECK:STDOUT:   %Base.F => constants.%Base.F
// CHECK:STDOUT:   %Base.F.specific_fn.loc6_1.2 => constants.%Base.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f0
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
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.type.0f1: type = fn_type @Base.F, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.F.cd3: %Base.F.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Base.F.specific_fn.f8c: <specific function> = specific_function %Base.F.cd3, @Base.F(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %pattern_type.2f0: type = pattern_type %Base.d0c [symbolic]
// CHECK:STDOUT:   %Base.d5d: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.type.cb1: type = fn_type @Base.F, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.F.eab: %Base.F.type.cb1 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.64f: type = pattern_type %Base.d5d [concrete]
// CHECK:STDOUT:   %Base.F.specific_fn.cd5: <specific function> = specific_function %Base.F.eab, @Base.F(%T1) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.ae5: ref %ptr.454 = class_element_access file.%v.var, element0 [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr.454 = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(%T1) [concrete]
// CHECK:STDOUT:   %Base.val: %Base.d5d = struct_value (%Base.vtable_ptr) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base: %Base.type = import_ref Main//generic_lib, Base, loaded [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//generic_lib, loc4_18, loaded [symbolic = @Base.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.462: ref %ptr.454 = import_ref Main//generic_lib, loc6_1, loaded [concrete = constants.%Base.vtable_decl]
// CHECK:STDOUT:   %Main.import_ref.05e: <witness> = import_ref Main//generic_lib, loc6_1, loaded [concrete = constants.%complete_type]
// CHECK:STDOUT:   %Main.import_ref.2af = import_ref Main//generic_lib, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.1cf = import_ref Main//generic_lib, loc5_37, unloaded
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//generic_lib, loc4_18, loaded [symbolic = @Base.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.4f0: <specific function> = import_ref Main//generic_lib, loc6_1, loaded [symbolic = constants.%Base.F.specific_fn.f8c]
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
// CHECK:STDOUT:     %v.patt: %pattern_type.64f = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.64f = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base.d5d = var %v.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %Base [concrete = constants.%Base.d5d] {
// CHECK:STDOUT:     %Base.ref: %Base.type = name_ref Base, imports.%Main.Base [concrete = constants.%Base.generic]
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, %T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.d5d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base.d5d = ref_binding v, %v.var [concrete = %v.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1;
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(imports.%Main.import_ref.b3bc94.2: type) [from "generic_lib.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type: type = fn_type @Base.F, @Base(%T) [symbolic = %Base.F.type (constants.%Base.F.type.0f1)]
// CHECK:STDOUT:   %Base.F: @Base.%Base.F.type (%Base.F.type.0f1) = struct_value () [symbolic = %Base.F (constants.%Base.F.cd3)]
// CHECK:STDOUT:   %Base.F.specific_fn: <specific function> = specific_function %Base.F, @Base.F(%T) [symbolic = %Base.F.specific_fn (constants.%Base.F.specific_fn.f8c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.05e
// CHECK:STDOUT:     vtable_decl = imports.%Main.import_ref.462
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.2af
// CHECK:STDOUT:     .F = imports.%Main.import_ref.1cf
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   imports.%Main.import_ref.4f0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @Base.F(imports.%Main.import_ref.b3bc94.1: type) [from "generic_lib.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [symbolic = %pattern_type (constants.%pattern_type.2f0)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %Base [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc7_20.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc7_20.2: ref %ptr.454 = class_element_access file.%v.var, element0 [concrete = constants.%.ae5]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable, @Base(constants.%T1) [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc7_20.3: init %ptr.454 to %.loc7_20.2 = in_place_init %Base.vtable_ptr [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc7_20.4: init %Base.d5d to file.%v.var = class_init (%.loc7_20.3) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc7_1: init %Base.d5d = converted %.loc7_20.1, %.loc7_20.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign file.%v.var, %.loc7_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.0f1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.cd3
// CHECK:STDOUT:   %Base.F.specific_fn => constants.%Base.F.specific_fn.f8c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.d0c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2f0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Base.F.type => constants.%Base.F.type.cb1
// CHECK:STDOUT:   %Base.F => constants.%Base.F.eab
// CHECK:STDOUT:   %Base.F.specific_fn => constants.%Base.F.specific_fn.cd5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base.F(constants.%T1) {
// CHECK:STDOUT:   %T => constants.%T1
// CHECK:STDOUT:   %Base => constants.%Base.d5d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.64f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_derived_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %G1: type = symbolic_binding G1, 0 [symbolic]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1.6d4458.1: type = class_type @T1, @T1(%G1) [symbolic]
// CHECK:STDOUT:   %pattern_type.253617.1: type = pattern_type %T1.6d4458.1 [symbolic]
// CHECK:STDOUT:   %T1.F.type.e39697.1: type = fn_type @T1.F, @T1(%G1) [symbolic]
// CHECK:STDOUT:   %T1.F.9b864d.1: %T1.F.type.e39697.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.F.specific_fn.46dc79.1: <specific function> = specific_function %T1.F.9b864d.1, @T1.F(%G1) [symbolic]
// CHECK:STDOUT:   %T1.vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.6d26d2.1: <witness> = require_complete_type %T1.6d4458.1 [symbolic]
// CHECK:STDOUT:   %G2: type = symbolic_binding G2, 0 [symbolic]
// CHECK:STDOUT:   %T2.type: type = generic_class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.generic: %T2.type = struct_value () [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2, @T2(%G2) [symbolic]
// CHECK:STDOUT:   %T1.6d4458.2: type = class_type @T1, @T1(%G2) [symbolic]
// CHECK:STDOUT:   %T1.F.type.e39697.2: type = fn_type @T1.F, @T1(%G2) [symbolic]
// CHECK:STDOUT:   %T1.F.9b864d.2: %T1.F.type.e39697.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.253617.2: type = pattern_type %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T1.F.specific_fn.46dc79.2: <specific function> = specific_function %T1.F.9b864d.2, @T1.F(%G2) [symbolic]
// CHECK:STDOUT:   %require_complete.6d26d2.2: <witness> = require_complete_type %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T2.vtable_decl: ref %ptr.454 = vtable_decl @T2.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1.6d4458.2} [symbolic]
// CHECK:STDOUT:   %complete_type.fc6: <witness> = complete_type_witness %struct_type.base [symbolic]
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
// CHECK:STDOUT:   %T1.decl: %T1.type = class_decl @T1 [concrete = constants.%T1.generic] {
// CHECK:STDOUT:     %G1.patt: %pattern_type.98f = symbolic_binding_pattern G1, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G1.loc4_17.2: type = symbolic_binding G1, 0 [symbolic = %G1.loc4_17.1 (constants.%G1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T2.decl: %T2.type = class_decl @T2 [concrete = constants.%T2.generic] {
// CHECK:STDOUT:     %G2.patt: %pattern_type.98f = symbolic_binding_pattern G2, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G2.loc8_12.2: type = symbolic_binding G2, 0 [symbolic = %G2.loc8_12.1 (constants.%G2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%G1.loc4_17.2: type) {
// CHECK:STDOUT:   %G1.loc4_17.1: type = symbolic_binding G1, 0 [symbolic = %G1.loc4_17.1 (constants.%G1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%G1.loc4_17.1) [symbolic = %T1.F.type (constants.%T1.F.type.e39697.1)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type.e39697.1) = struct_value () [symbolic = %T1.F (constants.%T1.F.9b864d.1)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2: <specific function> = specific_function %T1.F, @T1.F(%G1.loc4_17.1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn.46dc79.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type.e39697.1) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F.9b864d.1)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type (%pattern_type.253617.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type (%pattern_type.253617.1) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1.6d4458.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1.6d4458.1)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%T1.6d4458.1, @T1(constants.%G1) [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1.6d4458.1) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T1.F.specific_fn.loc6_1.1: <specific function> = specific_function %T1.F.decl, @T1.F(constants.%G1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn.46dc79.1)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete = constants.%T1.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1.6d4458.1
// CHECK:STDOUT:     .F = %T1.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T2(%G2.loc8_12.2: type) {
// CHECK:STDOUT:   %G2.loc8_12.1: type = symbolic_binding G2, 0 [symbolic = %G2.loc8_12.1 (constants.%G2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.loc9_21.2: type = class_type @T1, @T1(%G2.loc8_12.1) [symbolic = %T1.loc9_21.2 (constants.%T1.6d4458.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1.loc9_21.2 [symbolic = %require_complete (constants.%require_complete.6d26d2.2)]
// CHECK:STDOUT:   %T2: type = class_type @T2, @T2(%G2.loc8_12.1) [symbolic = %T2 (constants.%T2)]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1.loc9_21.2 [symbolic = %T2.elem (constants.%T2.elem)]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%G2.loc8_12.1) [symbolic = %T1.F.type (constants.%T1.F.type.e39697.2)]
// CHECK:STDOUT:   %T1.F: @T2.%T1.F.type (%T1.F.type.e39697.2) = struct_value () [symbolic = %T1.F (constants.%T1.F.9b864d.2)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc10_1.2: <specific function> = specific_function %T1.F, @T1.F(%G2.loc8_12.1) [symbolic = %T1.F.specific_fn.loc10_1.2 (constants.%T1.F.specific_fn.46dc79.2)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @T2.%T1.loc9_21.2 (%T1.6d4458.2)} [symbolic = %struct_type.base (constants.%struct_type.base)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.fc6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.ref: %T1.type = name_ref T1, file.%T1.decl [concrete = constants.%T1.generic]
// CHECK:STDOUT:     %G2.ref: type = name_ref G2, %G2.loc8_12.2 [symbolic = %G2.loc8_12.1 (constants.%G2)]
// CHECK:STDOUT:     %T1.loc9_21.1: type = class_type @T1, @T1(constants.%G2) [symbolic = %T1.loc9_21.2 (constants.%T1.6d4458.2)]
// CHECK:STDOUT:     %.loc9: @T2.%T2.elem (%T2.elem) = base_decl %T1.loc9_21.1, element0 [concrete]
// CHECK:STDOUT:     %T1.F.specific_fn.loc10_1.1: <specific function> = specific_function constants.%T1.F.9b864d.2, @T1.F(constants.%G2) [symbolic = %T1.F.specific_fn.loc10_1.2 (constants.%T1.F.specific_fn.46dc79.2)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T2.vtable [concrete = constants.%T2.vtable_decl]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.fc6)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T2
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:     .G2 = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     extend %T1.loc9_21.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T2.vtable {
// CHECK:STDOUT:   @T2.%T1.F.specific_fn.loc10_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%G1.loc4_17.2: type) {
// CHECK:STDOUT:   %G1: type = symbolic_binding G1, 0 [symbolic = %G1 (constants.%G1)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%G1) [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.253617.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic = %require_complete (constants.%require_complete.6d26d2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @T1.F.%T1 (%T1.6d4458.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%G1) {
// CHECK:STDOUT:   %G1.loc4_17.1 => constants.%G1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type.e39697.1
// CHECK:STDOUT:   %T1.F => constants.%T1.F.9b864d.1
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn.46dc79.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%G1) {
// CHECK:STDOUT:   %G1 => constants.%G1
// CHECK:STDOUT:   %T1 => constants.%T1.6d4458.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253617.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6d26d2.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%G2) {
// CHECK:STDOUT:   %G2.loc8_12.1 => constants.%G2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%G2) {
// CHECK:STDOUT:   %G1.loc4_17.1 => constants.%G2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type.e39697.2
// CHECK:STDOUT:   %T1.F => constants.%T1.F.9b864d.2
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn.46dc79.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%G2) {
// CHECK:STDOUT:   %G1 => constants.%G2
// CHECK:STDOUT:   %T1 => constants.%T1.6d4458.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253617.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6d26d2.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_derived_generic_context.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %G1: type = symbolic_binding G1, 0 [symbolic]
// CHECK:STDOUT:   %T1.type: type = generic_class_type @T1 [concrete]
// CHECK:STDOUT:   %T1.generic: %T1.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1.6d4458.1: type = class_type @T1, @T1(%G1) [symbolic]
// CHECK:STDOUT:   %pattern_type.253617.1: type = pattern_type %T1.6d4458.1 [symbolic]
// CHECK:STDOUT:   %T1.F.type.e39697.1: type = fn_type @T1.F, @T1(%G1) [symbolic]
// CHECK:STDOUT:   %T1.F.9b864d.1: %T1.F.type.e39697.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %T1.F.specific_fn.46dc79.1: <specific function> = specific_function %T1.F.9b864d.1, @T1.F(%G1) [symbolic]
// CHECK:STDOUT:   %T1.vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %require_complete.6d26d2.1: <witness> = require_complete_type %T1.6d4458.1 [symbolic]
// CHECK:STDOUT:   %G2: type = symbolic_binding G2, 0 [symbolic]
// CHECK:STDOUT:   %T2.type: type = generic_class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.generic: %T2.type = struct_value () [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2, @T2(%G2) [symbolic]
// CHECK:STDOUT:   %T3: type = class_type @T3, @T3(%G2) [symbolic]
// CHECK:STDOUT:   %T1.6d4458.2: type = class_type @T1, @T1(%G2) [symbolic]
// CHECK:STDOUT:   %T1.F.type.e39697.2: type = fn_type @T1.F, @T1(%G2) [symbolic]
// CHECK:STDOUT:   %T1.F.9b864d.2: %T1.F.type.e39697.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.253617.2: type = pattern_type %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T1.F.specific_fn.46dc79.2: <specific function> = specific_function %T1.F.9b864d.2, @T1.F(%G2) [symbolic]
// CHECK:STDOUT:   %require_complete.6d26d2.2: <witness> = require_complete_type %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T3.elem: type = unbound_element_type %T3, %T1.6d4458.2 [symbolic]
// CHECK:STDOUT:   %T3.vtable_decl: ref %ptr.454 = vtable_decl @T3.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %T1.6d4458.2} [symbolic]
// CHECK:STDOUT:   %complete_type.fc6: <witness> = complete_type_witness %struct_type.base [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
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
// CHECK:STDOUT:   %T1.decl: %T1.type = class_decl @T1 [concrete = constants.%T1.generic] {
// CHECK:STDOUT:     %G1.patt: %pattern_type.98f = symbolic_binding_pattern G1, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G1.loc4_17.2: type = symbolic_binding G1, 0 [symbolic = %G1.loc4_17.1 (constants.%G1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %T2.decl: %T2.type = class_decl @T2 [concrete = constants.%T2.generic] {
// CHECK:STDOUT:     %G2.patt: %pattern_type.98f = symbolic_binding_pattern G2, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_15.1: type = splice_block %.loc8_15.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_15.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G2.loc8_12.2: type = symbolic_binding G2, 0 [symbolic = %G2.loc8_12.1 (constants.%G2)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%G1.loc4_17.2: type) {
// CHECK:STDOUT:   %G1.loc4_17.1: type = symbolic_binding G1, 0 [symbolic = %G1.loc4_17.1 (constants.%G1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%G1.loc4_17.1) [symbolic = %T1.F.type (constants.%T1.F.type.e39697.1)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type.e39697.1) = struct_value () [symbolic = %T1.F (constants.%T1.F.9b864d.1)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2: <specific function> = specific_function %T1.F, @T1.F(%G1.loc4_17.1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn.46dc79.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type.e39697.1) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F.9b864d.1)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type (%pattern_type.253617.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type (%pattern_type.253617.1) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1.6d4458.1) = value_param call_param0
// CHECK:STDOUT:       %.loc5_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1.6d4458.1)] {
// CHECK:STDOUT:         %.loc5_29.2: type = specific_constant constants.%T1.6d4458.1, @T1(constants.%G1) [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc5_29.2 [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1.6d4458.1) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T1.F.specific_fn.loc6_1.1: <specific function> = specific_function %T1.F.decl, @T1.F(constants.%G1) [symbolic = %T1.F.specific_fn.loc6_1.2 (constants.%T1.F.specific_fn.46dc79.1)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T1.vtable [concrete = constants.%T1.vtable_decl]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T1.6d4458.1
// CHECK:STDOUT:     .F = %T1.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T2(%G2.loc8_12.2: type) {
// CHECK:STDOUT:   %G2.loc8_12.1: type = symbolic_binding G2, 0 [symbolic = %G2.loc8_12.1 (constants.%G2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T3: type = class_type @T3, @T3(%G2.loc8_12.1) [symbolic = %T3 (constants.%T3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T3.decl: type = class_decl @T3 [symbolic = @T2.%T3 (constants.%T3)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T2
// CHECK:STDOUT:     .T3 = %T3.decl
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:     .G2 = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T3(@T2.%G2.loc8_12.2: type) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G2: type = symbolic_binding G2, 0 [symbolic = %G2 (constants.%G2)]
// CHECK:STDOUT:   %T1.loc10_24.2: type = class_type @T1, @T1(%G2) [symbolic = %T1.loc10_24.2 (constants.%T1.6d4458.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1.loc10_24.2 [symbolic = %require_complete (constants.%require_complete.6d26d2.2)]
// CHECK:STDOUT:   %T3: type = class_type @T3, @T3(%G2) [symbolic = %T3 (constants.%T3)]
// CHECK:STDOUT:   %T3.elem: type = unbound_element_type %T3, %T1.loc10_24.2 [symbolic = %T3.elem (constants.%T3.elem)]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%G2) [symbolic = %T1.F.type (constants.%T1.F.type.e39697.2)]
// CHECK:STDOUT:   %T1.F: @T3.%T1.F.type (%T1.F.type.e39697.2) = struct_value () [symbolic = %T1.F (constants.%T1.F.9b864d.2)]
// CHECK:STDOUT:   %T1.F.specific_fn.loc11_3.2: <specific function> = specific_function %T1.F, @T1.F(%G2) [symbolic = %T1.F.specific_fn.loc11_3.2 (constants.%T1.F.specific_fn.46dc79.2)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @T3.%T1.loc10_24.2 (%T1.6d4458.2)} [symbolic = %struct_type.base (constants.%struct_type.base)]
// CHECK:STDOUT:   %complete_type.loc11_3.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc11_3.2 (constants.%complete_type.fc6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.ref: %T1.type = name_ref T1, file.%T1.decl [concrete = constants.%T1.generic]
// CHECK:STDOUT:     %G2.ref: type = name_ref G2, @T2.%G2.loc8_12.2 [symbolic = %G2 (constants.%G2)]
// CHECK:STDOUT:     %T1.loc10_24.1: type = class_type @T1, @T1(constants.%G2) [symbolic = %T1.loc10_24.2 (constants.%T1.6d4458.2)]
// CHECK:STDOUT:     %.loc10: @T3.%T3.elem (%T3.elem) = base_decl %T1.loc10_24.1, element0 [concrete]
// CHECK:STDOUT:     %T1.F.specific_fn.loc11_3.1: <specific function> = specific_function constants.%T1.F.9b864d.2, @T1.F(constants.%G2) [symbolic = %T1.F.specific_fn.loc11_3.2 (constants.%T1.F.specific_fn.46dc79.2)]
// CHECK:STDOUT:     %vtable_decl: ref %ptr.454 = vtable_decl @T3.vtable [concrete = constants.%T3.vtable_decl]
// CHECK:STDOUT:     %complete_type.loc11_3.1: <witness> = complete_type_witness constants.%struct_type.base [symbolic = %complete_type.loc11_3.2 (constants.%complete_type.fc6)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_3.1
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%T3
// CHECK:STDOUT:     .T1 = <poisoned>
// CHECK:STDOUT:     .G2 = <poisoned>
// CHECK:STDOUT:     .base = %.loc10
// CHECK:STDOUT:     extend %T1.loc10_24.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T1.vtable {
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @T3.vtable {
// CHECK:STDOUT:   @T3.%T1.F.specific_fn.loc11_3.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%G1.loc4_17.2: type) {
// CHECK:STDOUT:   %G1: type = symbolic_binding G1, 0 [symbolic = %G1 (constants.%G1)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%G1) [symbolic = %T1 (constants.%T1.6d4458.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.253617.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic = %require_complete (constants.%require_complete.6d26d2.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn(%self.param: @T1.F.%T1 (%T1.6d4458.1)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%G1) {
// CHECK:STDOUT:   %G1.loc4_17.1 => constants.%G1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type.e39697.1
// CHECK:STDOUT:   %T1.F => constants.%T1.F.9b864d.1
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn.46dc79.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%G1) {
// CHECK:STDOUT:   %G1 => constants.%G1
// CHECK:STDOUT:   %T1 => constants.%T1.6d4458.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253617.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6d26d2.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T2(constants.%G2) {
// CHECK:STDOUT:   %G2.loc8_12.1 => constants.%G2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T3(constants.%G2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1(constants.%G2) {
// CHECK:STDOUT:   %G1.loc4_17.1 => constants.%G2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type => constants.%T1.F.type.e39697.2
// CHECK:STDOUT:   %T1.F => constants.%T1.F.9b864d.2
// CHECK:STDOUT:   %T1.F.specific_fn.loc6_1.2 => constants.%T1.F.specific_fn.46dc79.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%G2) {
// CHECK:STDOUT:   %G1 => constants.%G2
// CHECK:STDOUT:   %T1 => constants.%T1.6d4458.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.253617.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6d26d2.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
