// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/method/virtual.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/method/virtual.carbon

// --- modifiers.carbon

package Modifiers;

//@dump-sem-ir-begin
base class Base {
  virtual fn H[self: Self]();
}

abstract class Abstract {
  abstract fn J[self: Self]();

  virtual fn K[self: Self]();
}
//@dump-sem-ir-end

// --- override_import.carbon

library "[[@TEST_NAME]]";

import Modifiers;

//@dump-sem-ir-begin
class Derived {
  extend base: Modifiers.Base;
  override fn H[self: Self]();
}
//@dump-sem-ir-end

fn Use() {
  //@dump-sem-ir-begin
  var unused d: Derived = {.base = {}};
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  var unused v: Modifiers.Base = {};
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  var b1: Base = {.m2 = i, .m1 = i};
  var unused b2: Base = {.m2 = 3, .m1 = 5};
  //@dump-sem-ir-end

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
  //@dump-sem-ir-begin
  override fn F[self: Self]();
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  override fn F[self: Self]();
  //@dump-sem-ir-end
}

// --- fail_impl_mismatch.carbon

library "[[@TEST_NAME]]";

base class Base {
  virtual fn F[self: Self]();
}

class Derived {
  extend base: Base;
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE+10]]:3: error: 0 arguments passed to function expecting 1 argument [CallArgCountMismatch]
  // CHECK:STDERR:   override fn F[self: Self](v: i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE+7]]:3: note: calling function declared here [InCallToEntity]
  // CHECK:STDERR:   override fn F[self: Self](v: i32);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE-11]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[self: Self](v: i32);
}

// --- impl_conversion.carbon

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
  //@dump-sem-ir-begin
  override fn F[self: Self]() -> T2;
  //@dump-sem-ir-end
}

fn Derived.F[unused self: Self]() -> T2 {
  return {};
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
  // CHECK:STDERR: fail_ref_self_mismatch.carbon:[[@LINE+10]]:3: error: value expression passed to reference parameter [ValueForRefParam]
  // CHECK:STDERR:   override fn F1[ref self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_ref_self_mismatch.carbon:[[@LINE+7]]:18: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn F1[ref self: Self]();
  // CHECK:STDERR:                  ^~~~~~~~~~~~~~
  // CHECK:STDERR: fail_ref_self_mismatch.carbon:[[@LINE-11]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual fn F1[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  //@dump-sem-ir-begin
  virtual fn F[unused self: Self]() { }
  //@dump-sem-ir-end
}

// --- with_dependent_arg.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  //@dump-sem-ir-begin
  virtual fn F[unused self: Self](unused t: T) { }
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  override fn F[unused self: Self](unused t: Base(T1)*) { }
  //@dump-sem-ir-end
}

// --- fail_impl_generic_specifically_mismatch.carbon

library "[[@TEST_NAME]]";

class T1;
class T2;

base class Base(T:! type) {
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE+6]]:42: error: cannot implicitly convert expression of type `T1*` to `T2*` [ConversionFailure]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T1*) { }
  // CHECK:STDERR:                                          ^~~~~~
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE+3]]:42: note: type `T1*` does not implement interface `Core.ImplicitAs(T2*)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T1*) { }
  // CHECK:STDERR:                                          ^~~~~~
  virtual fn F[unused self: Self](unused t: T1*) { }
}

class D1 {
  extend base: Base(T1);
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE+7]]:43: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn F[unused self: Self](unused t: T2*) { }
  // CHECK:STDERR:                                           ^~~~~~
  // CHECK:STDERR: fail_impl_generic_specifically_mismatch.carbon:[[@LINE-8]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T1*) { }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  override fn F[unused self: Self](unused t: T2*) { }
}

// --- fail_impl_generic_generic_mismatch.carbon

library "[[@TEST_NAME]]";

abstract class Base(T:! type) {
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE+6]]:42: error: cannot implicitly convert expression of type `T*` to `T` [ConversionFailure]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T*) { }
  // CHECK:STDERR:                                          ^~~~~
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE+3]]:42: note: type `T*` does not implement interface `Core.ImplicitAs(T)` [MissingImplInMemberAccessInContext]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T*) { }
  // CHECK:STDERR:                                          ^~~~~
  virtual fn F[unused self: Self](unused t: T*) { }
}
class Derived(T:! type) {
  extend base: Base(T);
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE+7]]:43: note: initializing function parameter [InCallToFunctionParam]
  // CHECK:STDERR:   override fn F[unused self: Self](unused t: T) { }
  // CHECK:STDERR:                                           ^~~~
  // CHECK:STDERR: fail_impl_generic_generic_mismatch.carbon:[[@LINE-7]]:3: note: while building thunk to match the signature of this function [ThunkSignature]
  // CHECK:STDERR:   virtual fn F[unused self: Self](unused t: T*) { }
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
  //@dump-sem-ir-begin
  override fn F[unused self: Self](unused t: T*) { }
  //@dump-sem-ir-end
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
    extend base: T1(G2);
  }
}

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.b8d: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %Base.vtable_decl: ref %ptr = vtable_decl @Base.vtable [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.0e2: type = pattern_type %Abstract [concrete]
// CHECK:STDOUT:   %Abstract.J.type: type = fn_type @Abstract.J [concrete]
// CHECK:STDOUT:   %Abstract.J: %Abstract.J.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract.K.type: type = fn_type @Abstract.K [concrete]
// CHECK:STDOUT:   %Abstract.K: %Abstract.K.type = struct_value () [concrete]
// CHECK:STDOUT:   %Abstract.vtable_decl: ref %ptr = vtable_decl @Abstract.vtable [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [concrete = constants.%Abstract] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %Base.H.decl: %Base.H.type = fn_decl @Base.H [concrete = constants.%Base.H] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.b8d = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.b8d = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0e2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.0e2 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Abstract.K.decl: %Abstract.K.type = fn_decl @Abstract.K [concrete = constants.%Abstract.K] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.0e2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.0e2 = at_binding_pattern self, %self.param_patt [concrete]
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
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.H.type.1f42d4.1: type = fn_type @Derived.H.loc9_30.1 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived.H.a6796b.1: %Derived.H.type.1f42d4.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.H.type.1f42d4.2: type = fn_type @Derived.H.loc9_30.2 [concrete]
// CHECK:STDOUT:   %Derived.H.a6796b.2: %Derived.H.type.1f42d4.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete]
// CHECK:STDOUT:   %struct_type.base.507: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.d6b: <witness> = complete_type_witness %struct_type.base.507 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct.6b1: %struct_type.base.f5e = struct_value (%empty_struct) [concrete]
// CHECK:STDOUT:   %.f93: type = partial_type %Base [concrete]
// CHECK:STDOUT:   %uninit: %ptr.454 = uninitialized_value [concrete]
// CHECK:STDOUT:   %struct.7da: %.f93 = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%uninit) [concrete]
// CHECK:STDOUT:   %Derived.val: %Derived = struct_value (%Base.val) [concrete]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc15_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Base.H.decl: %Base.H.type = fn_decl @Base.H [concrete = constants.%Base.H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %Derived.H.decl.loc9_30.1: %Derived.H.type.1f42d4.1 = fn_decl @Derived.H.loc9_30.1 [concrete = constants.%Derived.H.a6796b.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.746 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.746 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.H.decl.loc9_30.2: %Derived.H.type.1f42d4.2 = fn_decl @Derived.H.loc9_30.2 [concrete = constants.%Derived.H.a6796b.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %vtable_decl: ref %ptr.454 = vtable_decl @Derived.vtable [concrete = constants.%Derived.vtable_decl]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.507 [concrete = constants.%complete_type.d6b]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Modifiers = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .H [hidden] = %Derived.H.decl.loc9_30.1
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.H.decl.loc9_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.H.loc9_30.1(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.H.loc9_30.2(%self.param: %Base) [thunk @Derived.%Derived.H.decl.loc9_30.1 for imports.%Base.H.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %H.ref: %Derived.H.type.1f42d4.1 = name_ref H, @Derived.%Derived.H.decl.loc9_30.1 [concrete = constants.%Derived.H.a6796b.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc9_30.1: ref %Derived = enclosing_class_access %.1, element0
// CHECK:STDOUT:   %.loc9_30.2: %Derived = acquire_value %.loc9_30.1
// CHECK:STDOUT:   %Derived.H.bound: <bound method> = bound_method %.loc9_30.2, %H.ref
// CHECK:STDOUT:   %Derived.H.call: init %empty_tuple.type = call %Derived.H.bound(%.loc9_30.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.746 = ref_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.746 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %Derived = var %d.var_patt
// CHECK:STDOUT:   %.loc15_37.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc15_38.1: %struct_type.base.f5e = struct_literal (%.loc15_37.1) [concrete = constants.%struct.6b1]
// CHECK:STDOUT:   %.loc15_38.2: ref %.f93 = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc15_37.2: ref %ptr.454 = class_element_access %.loc15_38.2, element0
// CHECK:STDOUT:   %uninit: %ptr.454 = uninitialized_value [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc15_37.3: init %ptr.454 to %.loc15_37.2 = in_place_init %uninit [concrete = constants.%uninit]
// CHECK:STDOUT:   %.loc15_37.4: init %.f93 to %.loc15_38.2 = class_init (%.loc15_37.3) [concrete = constants.%struct.7da]
// CHECK:STDOUT:   %.loc15_38.3: init %.f93 = converted %.loc15_37.1, %.loc15_37.4 [concrete = constants.%struct.7da]
// CHECK:STDOUT:   %.loc15_38.4: init %Base = as_compatible %.loc15_38.3 [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc15_38.5: init %Derived to %d.var = class_init (%.loc15_38.4) [concrete = constants.%Derived.val]
// CHECK:STDOUT:   %Derived.vtable_ptr: ref %ptr.454 = vtable_ptr @Derived.vtable [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %.loc15_38.6: ref %Base = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc15_38.7: ref %ptr.454 = class_element_access %.loc15_38.6, element0
// CHECK:STDOUT:   %.loc15_38.8: init %ptr.454 to %.loc15_38.7 = in_place_init %Derived.vtable_ptr [concrete = constants.%Derived.vtable_ptr]
// CHECK:STDOUT:   %.loc15_38.9: init %Derived = update_init %.loc15_38.5, %.loc15_38.8
// CHECK:STDOUT:   %.loc15_3: init %Derived = converted %.loc15_38.1, %.loc15_38.9
// CHECK:STDOUT:   assign %d.var, %.loc15_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d: ref %Derived = ref_binding d, %d.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %d.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%d.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.3(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.4(%self.param: ref %struct_type.base.507) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.5(%self.param: ref %Derived) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %pattern_type.b8d: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.H.type: type = fn_type @Base.H [concrete]
// CHECK:STDOUT:   %Base.H: %Base.H.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.3: type = fn_type @Destroy.Op.loc8_3.3 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.3: %Destroy.Op.type.1d8f74.3 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   constants.%Base.H
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.b8d = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.b8d = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base = var %v.var_patt
// CHECK:STDOUT:   %.loc8_35.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc8_35.2: ref %ptr.454 = class_element_access %v.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc8_35.3: init %ptr.454 to %.loc8_35.2 = in_place_init %Base.vtable_ptr [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc8_35.4: init %Base to %v.var = class_init (%.loc8_35.3) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc8_3: init %Base = converted %.loc8_35.1, %.loc8_35.4 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %v.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_26: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = ref_binding v, %v.var
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.1a2547.3
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.2(%self.param: ref %struct_type.vptr) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc8_3.3(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.912: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.763: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_3.410: %i32 = int_value 3 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.810: type = struct_type {.m2: %i32, .m1: %i32} [concrete]
// CHECK:STDOUT:   %Base.vtable_ptr: ref %ptr.454 = vtable_ptr @Base.vtable [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1.5f2: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.m2.m1.5f2 = struct_value (%int_3.1ba, %int_5.64b) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %bound_method.c9d: <bound method> = bound_method %int_5.64b, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_5.eef: %i32 = int_value 5 [concrete]
// CHECK:STDOUT:   %Base.val: %Base = struct_value (%Base.vtable_ptr, %int_5.eef, %int_3.410) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.5: type = fn_type @Destroy.Op.loc15_3.5 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.5: %Destroy.Op.type.1d8f74.5 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.912 = ref_binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.912 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %Base = var %b1.var_patt
// CHECK:STDOUT:   %i.ref.loc14_25: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc14_34: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.m2.m1.810 = struct_literal (%i.ref.loc14_25, %i.ref.loc14_34)
// CHECK:STDOUT:   %.loc14_35.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr.loc14: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc14_35.3: init %ptr.454 to %.loc14_35.2 = in_place_init %Base.vtable_ptr.loc14 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc14_34: %i32 = acquire_value %i.ref.loc14_34
// CHECK:STDOUT:   %impl.elem0.loc14_34: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_34.1: <bound method> = bound_method %.loc14_34, %impl.elem0.loc14_34
// CHECK:STDOUT:   %specific_fn.loc14_34: <specific function> = specific_function %impl.elem0.loc14_34, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_34.2: <bound method> = bound_method %.loc14_34, %specific_fn.loc14_34
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_34: init %i32 = call %bound_method.loc14_34.2(%.loc14_34)
// CHECK:STDOUT:   %.loc14_35.4: ref %i32 = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc14_35.5: init %i32 to %.loc14_35.4 = in_place_init %Int.as.Copy.impl.Op.call.loc14_34
// CHECK:STDOUT:   %.loc14_25: %i32 = acquire_value %i.ref.loc14_25
// CHECK:STDOUT:   %impl.elem0.loc14_25: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_25.1: <bound method> = bound_method %.loc14_25, %impl.elem0.loc14_25
// CHECK:STDOUT:   %specific_fn.loc14_25: <specific function> = specific_function %impl.elem0.loc14_25, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_25.2: <bound method> = bound_method %.loc14_25, %specific_fn.loc14_25
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc14_25: init %i32 = call %bound_method.loc14_25.2(%.loc14_25)
// CHECK:STDOUT:   %.loc14_35.6: ref %i32 = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc14_35.7: init %i32 to %.loc14_35.6 = in_place_init %Int.as.Copy.impl.Op.call.loc14_25
// CHECK:STDOUT:   %.loc14_35.8: init %Base to %b1.var = class_init (%.loc14_35.3, %.loc14_35.5, %.loc14_35.7)
// CHECK:STDOUT:   %.loc14_3: init %Base = converted %.loc14_35.1, %.loc14_35.8
// CHECK:STDOUT:   assign %b1.var, %.loc14_3
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = ref_binding b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.912 = ref_binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.912 = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var %b2.var_patt
// CHECK:STDOUT:   %int_3.loc15: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc15_42.1: %struct_type.m2.m1.5f2 = struct_literal (%int_3.loc15, %int_5) [concrete = constants.%struct]
// CHECK:STDOUT:   %.loc15_42.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %Base.vtable_ptr.loc15: ref %ptr.454 = vtable_ptr @Base.vtable [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %.loc15_42.3: init %ptr.454 to %.loc15_42.2 = in_place_init %Base.vtable_ptr.loc15 [concrete = constants.%Base.vtable_ptr]
// CHECK:STDOUT:   %impl.elem0.loc15_42.1: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc15_42.1: <bound method> = bound_method %int_5, %impl.elem0.loc15_42.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.f6c]
// CHECK:STDOUT:   %specific_fn.loc15_42.1: <specific function> = specific_function %impl.elem0.loc15_42.1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_42.2: <bound method> = bound_method %int_5, %specific_fn.loc15_42.1 [concrete = constants.%bound_method.c9d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_42.1: init %i32 = call %bound_method.loc15_42.2(%int_5) [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc15_42.4: init %i32 = converted %int_5, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_42.1 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %.loc15_42.5: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc15_42.6: init %i32 to %.loc15_42.5 = in_place_init %.loc15_42.4 [concrete = constants.%int_5.eef]
// CHECK:STDOUT:   %impl.elem0.loc15_42.2: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc15_42.3: <bound method> = bound_method %int_3.loc15, %impl.elem0.loc15_42.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.485]
// CHECK:STDOUT:   %specific_fn.loc15_42.2: <specific function> = specific_function %impl.elem0.loc15_42.2, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_42.4: <bound method> = bound_method %int_3.loc15, %specific_fn.loc15_42.2 [concrete = constants.%bound_method.763]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_42.2: init %i32 = call %bound_method.loc15_42.4(%int_3.loc15) [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc15_42.7: init %i32 = converted %int_3.loc15, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc15_42.2 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc15_42.8: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc15_42.9: init %i32 to %.loc15_42.8 = in_place_init %.loc15_42.7 [concrete = constants.%int_3.410]
// CHECK:STDOUT:   %.loc15_42.10: init %Base to %b2.var = class_init (%.loc15_42.3, %.loc15_42.6, %.loc15_42.9) [concrete = constants.%Base.val]
// CHECK:STDOUT:   %.loc15_3: init %Base = converted %.loc15_42.1, %.loc15_42.10 [concrete = constants.%Base.val]
// CHECK:STDOUT:   assign %b2.var, %.loc15_3
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = ref_binding b2, %b2.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Destroy.Op.bound.loc15: <bound method> = bound_method %b2.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc15: init %empty_tuple.type = call %Destroy.Op.bound.loc15(%b2.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc14: <bound method> = bound_method %b1.var, constants.%Destroy.Op.1a2547.5
// CHECK:STDOUT:   %Destroy.Op.call.loc14: init %empty_tuple.type = call %Destroy.Op.bound.loc14(%b1.var)
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.1(%self.param: ref %ptr.454) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.2(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.3(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.4(%self.param: ref %struct_type.vptr.m1.m2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.5(%self.param: ref %Base) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractBase: type = class_type @AbstractBase [concrete]
// CHECK:STDOUT:   %AbstractBase.F.type: type = fn_type @AbstractBase.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %AbstractBase.F: %AbstractBase.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %AbstractIntermediate: type = class_type @AbstractIntermediate [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.1: type = fn_type @Derived.F.loc15_30.1 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.1: %Derived.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.2: type = fn_type @Derived.F.loc15_30.2 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.2: %Derived.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.decl.loc15_30.1: %Derived.F.type.48cdab.1 = fn_decl @Derived.F.loc15_30.1 [concrete = constants.%Derived.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.746 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.746 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.F.decl.loc15_30.2: %Derived.F.type.48cdab.2 = fn_decl @Derived.F.loc15_30.2 [concrete = constants.%Derived.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .AbstractIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F [hidden] = %Derived.F.decl.loc15_30.1
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
// CHECK:STDOUT:   @Derived.%Derived.F.decl.loc15_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F.loc15_30.1(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F.loc15_30.2(%self.param: %AbstractBase) [thunk @Derived.%Derived.F.decl.loc15_30.1 for @AbstractBase.%AbstractBase.F.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %Derived.F.type.48cdab.1 = name_ref F, @Derived.%Derived.F.decl.loc15_30.1 [concrete = constants.%Derived.F.bd5be1.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_30.1: ref %AbstractIntermediate = enclosing_class_access %.loc5, element0
// CHECK:STDOUT:   %.loc15_30.2: ref %Derived = enclosing_class_access %.loc15_30.1, element0
// CHECK:STDOUT:   %.loc15_30.3: %Derived = acquire_value %.loc15_30.2
// CHECK:STDOUT:   %Derived.F.bound: <bound method> = bound_method %.loc15_30.3, %F.ref
// CHECK:STDOUT:   %Derived.F.call: init %empty_tuple.type = call %Derived.F.bound(%.loc15_30.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- virtual_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %VirtualBase: type = class_type @VirtualBase [concrete]
// CHECK:STDOUT:   %VirtualBase.F.type: type = fn_type @VirtualBase.F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %VirtualBase.F: %VirtualBase.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %VirtualIntermediate: type = class_type @VirtualIntermediate [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %pattern_type.746: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.1: type = fn_type @Derived.F.loc15_30.1 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.1: %Derived.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.2: type = fn_type @Derived.F.loc15_30.2 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.2: %Derived.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.decl.loc15_30.1: %Derived.F.type.48cdab.1 = fn_decl @Derived.F.loc15_30.1 [concrete = constants.%Derived.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.746 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.746 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = value_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.F.decl.loc15_30.2: %Derived.F.type.48cdab.2 = fn_decl @Derived.F.loc15_30.2 [concrete = constants.%Derived.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .VirtualIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F [hidden] = %Derived.F.decl.loc15_30.1
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
// CHECK:STDOUT:   @Derived.%Derived.F.decl.loc15_30.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F.loc15_30.1(%self.param: %Derived);
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F.loc15_30.2(%self.param: %VirtualBase) [thunk @Derived.%Derived.F.decl.loc15_30.1 for @VirtualBase.%VirtualBase.F.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %Derived.F.type.48cdab.1 = name_ref F, @Derived.%Derived.F.decl.loc15_30.1 [concrete = constants.%Derived.F.bd5be1.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc15_30.1: ref %VirtualIntermediate = enclosing_class_access %.loc5, element0
// CHECK:STDOUT:   %.loc15_30.2: ref %Derived = enclosing_class_access %.loc15_30.1, element0
// CHECK:STDOUT:   %.loc15_30.3: %Derived = acquire_value %.loc15_30.2
// CHECK:STDOUT:   %Derived.F.bound: <bound method> = bound_method %.loc15_30.3, %F.ref
// CHECK:STDOUT:   %Derived.F.call: init %empty_tuple.type = call %Derived.F.bound(%.loc15_30.3)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_conversion.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.a74: type = facet_type <@ImplicitAs, @ImplicitAs(%T1)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness @T2.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert.type: type = fn_type @T2.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert: %T2.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.a74 = facet_value %T2, (%ImplicitAs.impl_witness) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.00a: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T1, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %.617: Core.Form = init_form %T2 [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.1: type = fn_type @Derived.F.loc27 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.1: %Derived.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Derived.F.type.48cdab.2: type = fn_type @Derived.F.loc23 [concrete]
// CHECK:STDOUT:   %Derived.F.bd5be1.2: %Derived.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT:   %.fd2: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.00a, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc23_36.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.decl.loc23_36.1: %Derived.F.type.48cdab.1 = fn_decl @Derived.F.loc27 [concrete = constants.%Derived.F.bd5be1.1] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T2.ref.loc23: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %.loc23: Core.Form = init_form %T2.ref.loc23 [concrete = constants.%.617]
// CHECK:STDOUT:     %self.param.loc23: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc23: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self.loc23: %Derived = value_binding self, %self.param.loc23
// CHECK:STDOUT:     %return.param.loc23: ref %T2 = out_param call_param1
// CHECK:STDOUT:     %return.loc23: ref %T2 = return_slot %return.param.loc23
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.decl.loc23_36.2: %Derived.F.type.48cdab.2 = fn_decl @Derived.F.loc23 [concrete = constants.%Derived.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc21
// CHECK:STDOUT:   .T2 = <poisoned>
// CHECK:STDOUT:   .F [hidden] = %Derived.F.decl.loc23_36.1
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.decl.loc23_36.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @Derived.F.loc23(%self.param: %Base) -> out %return.param: %T1 [thunk @Derived.%Derived.F.decl.loc23_36.1 for @Base.%Base.F.decl] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %Derived.F.type.48cdab.1 = name_ref F, @Derived.%Derived.F.decl.loc23_36.1 [concrete = constants.%Derived.F.bd5be1.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc23_36.1: ref %Derived = enclosing_class_access %.loc17_20, element0
// CHECK:STDOUT:   %.loc23_36.2: %Derived = acquire_value %.loc23_36.1
// CHECK:STDOUT:   %Derived.F.bound: <bound method> = bound_method %.loc23_36.2, %F.ref
// CHECK:STDOUT:   %.loc23_36.3: ref %T2 = temporary_storage
// CHECK:STDOUT:   %Derived.F.call: init %T2 to %.loc23_36.3 = call %Derived.F.bound(%.loc23_36.2)
// CHECK:STDOUT:   %impl.elem0: %.fd2 = impl_witness_access constants.%ImplicitAs.impl_witness, element0 [concrete = constants.%T2.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %Derived.F.call, %impl.elem0
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc23_36.4: ref %T2 = temporary %.loc23_36.3, %Derived.F.call
// CHECK:STDOUT:   %.loc23_36.5: %T2 = acquire_value %.loc23_36.4
// CHECK:STDOUT:   %T2.as.ImplicitAs.impl.Convert.call: init %T1 to %.loc17_33 = call %bound_method(%.loc23_36.5)
// CHECK:STDOUT:   %.loc23_36.6: init %T1 = converted %Derived.F.call, %T2.as.ImplicitAs.impl.Convert.call
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.loc23_36.4, constants.%Destroy.Op.1a2547.2
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%.loc23_36.4)
// CHECK:STDOUT:   return %.loc23_36.6 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_36.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc23_36.2(%self.param: ref %T2) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- generic_with_virtual.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c79: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T1.F.specific_fn: <specific function> = specific_function %T1.F, @T1.F(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T1 [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.loc4_16.1) [symbolic = %T1.F.type (constants.%T1.F.type)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type) = struct_value () [symbolic = %T1.F (constants.%T1.F)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type (%pattern_type.c79) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type (%pattern_type.c79) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc6_29.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_29.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1) = value_binding self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc8_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T1 [symbolic = %pattern_type (constants.%pattern_type.c79)]
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
// CHECK:STDOUT:   %T1.F.specific_fn.loc8_1.2 => constants.%T1.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c79
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- with_dependent_arg.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.c79: type = pattern_type %T1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T) [symbolic]
// CHECK:STDOUT:   %T1.F: %T1.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %T1.F.specific_fn: <specific function> = specific_function %T1.F, @T1.F(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.ce3: <witness> = require_complete_type %T1 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @T1(%T.loc4_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T1.F.type: type = fn_type @T1.F, @T1(%T.loc4_16.1) [symbolic = %T1.F.type (constants.%T1.F.type)]
// CHECK:STDOUT:   %T1.F: @T1.%T1.F.type (%T1.F.type) = struct_value () [symbolic = %T1.F (constants.%T1.F)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T1.F.decl: @T1.%T1.F.type (%T1.F.type) = fn_decl @T1.F [symbolic = @T1.%T1.F (constants.%T1.F)] {
// CHECK:STDOUT:       %self.param_patt: @T1.F.%pattern_type.loc6_27 (%pattern_type.c79) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @T1.F.%pattern_type.loc6_27 (%pattern_type.c79) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @T1.F.%pattern_type.loc6_43 (%pattern_type.51d) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @T1.F.%pattern_type.loc6_43 (%pattern_type.51d) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @T1.F.%T1 (%T1) = value_param call_param0
// CHECK:STDOUT:       %.loc6_29.1: type = splice_block %Self.ref [symbolic = %T1 (constants.%T1)] {
// CHECK:STDOUT:         %.loc6_29.2: type = specific_constant constants.%T1, @T1(constants.%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc6_29.2 [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @T1.F.%T1 (%T1) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @T1.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @T1.%T.loc4_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @T1.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
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
// CHECK:STDOUT:   @T1.%T1.F.specific_fn.loc8_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic virtual fn @T1.F(@T1.%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T1: type = class_type @T1, @T1(%T) [symbolic = %T1 (constants.%T1)]
// CHECK:STDOUT:   %pattern_type.loc6_27: type = pattern_type %T1 [symbolic = %pattern_type.loc6_27 (constants.%pattern_type.c79)]
// CHECK:STDOUT:   %pattern_type.loc6_43: type = pattern_type %T [symbolic = %pattern_type.loc6_43 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27: <witness> = require_complete_type %T1 [symbolic = %require_complete.loc6_27 (constants.%require_complete.ce3)]
// CHECK:STDOUT:   %require_complete.loc6_43: <witness> = require_complete_type %T [symbolic = %require_complete.loc6_43 (constants.%require_complete.944)]
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
// CHECK:STDOUT:   %T1.F.specific_fn.loc8_1.2 => constants.%T1.F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @T1.F(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T1 => constants.%T1
// CHECK:STDOUT:   %pattern_type.loc6_27 => constants.%pattern_type.c79
// CHECK:STDOUT:   %pattern_type.loc6_43 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_27 => constants.%require_complete.ce3
// CHECK:STDOUT:   %require_complete.loc6_43 => constants.%require_complete.944
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_specifically.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %D1: type = class_type @D1 [concrete]
// CHECK:STDOUT:   %Base.5b2: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %ptr.2ee: type = ptr_type %Base.5b2 [concrete]
// CHECK:STDOUT:   %pattern_type.e0f: type = pattern_type %ptr.2ee [concrete]
// CHECK:STDOUT:   %pattern_type.c35: type = pattern_type %D1 [concrete]
// CHECK:STDOUT:   %D1.F.type.48cdab.1: type = fn_type @D1.F.loc11_57.1 [concrete]
// CHECK:STDOUT:   %D1.F.bd5be1.1: %D1.F.type.48cdab.1 = struct_value () [concrete]
// CHECK:STDOUT:   %D1.F.type.48cdab.2: type = fn_type @D1.F.loc11_57.2 [concrete]
// CHECK:STDOUT:   %D1.F.bd5be1.2: %D1.F.type.48cdab.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D1 {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %D1.F.decl.loc11_57.1: %D1.F.type.48cdab.1 = fn_decl @D1.F.loc11_57.1 [concrete = constants.%D1.F.bd5be1.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c35 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.c35 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %t.param_patt: %pattern_type.e0f = value_param_pattern [concrete]
// CHECK:STDOUT:     %t.patt: %pattern_type.e0f = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %D1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%D1 [concrete = constants.%D1]
// CHECK:STDOUT:     %self: %D1 = value_binding self, %self.param
// CHECK:STDOUT:     %t.param: %ptr.2ee = value_param call_param1
// CHECK:STDOUT:     %.loc11: type = splice_block %ptr [concrete = constants.%ptr.2ee] {
// CHECK:STDOUT:       %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:       %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:       %Base: type = class_type @Base, @Base(constants.%T1) [concrete = constants.%Base.5b2]
// CHECK:STDOUT:       %ptr: type = ptr_type %Base [concrete = constants.%ptr.2ee]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %t: %ptr.2ee = value_binding t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D1.F.decl.loc11_57.2: %D1.F.type.48cdab.2 = fn_decl @D1.F.loc11_57.2 [concrete = constants.%D1.F.bd5be1.2] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:   vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D1
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F [hidden] = %D1.F.decl.loc11_57.1
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @D1.vtable {
// CHECK:STDOUT:   @D1.%D1.F.decl.loc11_57.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @D1.F.loc11_57.1(%self.param: %D1, %t.param: %ptr.2ee) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: override fn @D1.F.loc11_57.2(%self.param: %Base.5b2, %t.param: %ptr.2ee) [thunk @D1.%D1.F.decl.loc11_57.1 for @Base.%Base.F.decl, @Base.F(constants.%T1)] {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %D1.F.type.48cdab.1 = name_ref F, @D1.%D1.F.decl.loc11_57.1 [concrete = constants.%D1.F.bd5be1.1]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc11_57.1: ref %D1 = enclosing_class_access %.loc5, element0
// CHECK:STDOUT:   %.loc11_57.2: %D1 = acquire_value %.loc11_57.1
// CHECK:STDOUT:   %D1.F.bound: <bound method> = bound_method %.loc11_57.2, %F.ref
// CHECK:STDOUT:   %D1.F.call: init %empty_tuple.type = call %D1.F.bound(%.loc11_57.2, %t.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_generic_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %Base.99a: type = class_type @Base, @Base(%ptr.e8f) [symbolic]
// CHECK:STDOUT:   %pattern_type.bdc: type = pattern_type %Base.99a [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.29f: <witness> = require_complete_type %Base.99a [symbolic]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.99a [symbolic]
// CHECK:STDOUT:   %pattern_type.839: type = pattern_type %Derived [symbolic]
// CHECK:STDOUT:   %Derived.F.type.77405e.1: type = fn_type @Derived.F.loc10_50.1, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.F.d85493.1: %Derived.F.type.77405e.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Derived.F.type.77405e.2: type = fn_type @Derived.F.loc10_50.2, @Derived(%T) [symbolic]
// CHECK:STDOUT:   %Derived.F.d85493.2: %Derived.F.type.77405e.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Derived.F.specific_fn.822c1a.1: <specific function> = specific_function %Derived.F.d85493.2, @Derived.F.loc10_50.2(%T) [symbolic]
// CHECK:STDOUT:   %struct_type.base.619: type = struct_type {.base: %Base.99a} [symbolic]
// CHECK:STDOUT:   %complete_type.db5: <witness> = complete_type_witness %struct_type.base.619 [symbolic]
// CHECK:STDOUT:   %require_complete.1e7: <witness> = require_complete_type %Derived [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Derived.F.specific_fn.822c1a.2: <specific function> = specific_function %Derived.F.d85493.1, @Derived.F.loc10_50.1(%T) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Derived(%T.loc7_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.type.loc10_50.1: type = fn_type @Derived.F.loc10_50.1, @Derived(%T.loc7_16.1) [symbolic = %Derived.F.type.loc10_50.1 (constants.%Derived.F.type.77405e.1)]
// CHECK:STDOUT:   %Derived.F.loc10_50.1: @Derived.%Derived.F.type.loc10_50.1 (%Derived.F.type.77405e.1) = struct_value () [symbolic = %Derived.F.loc10_50.1 (constants.%Derived.F.d85493.1)]
// CHECK:STDOUT:   %Derived.F.type.loc10_50.2: type = fn_type @Derived.F.loc10_50.2, @Derived(%T.loc7_16.1) [symbolic = %Derived.F.type.loc10_50.2 (constants.%Derived.F.type.77405e.2)]
// CHECK:STDOUT:   %Derived.F.loc10_50.2: @Derived.%Derived.F.type.loc10_50.2 (%Derived.F.type.77405e.2) = struct_value () [symbolic = %Derived.F.loc10_50.2 (constants.%Derived.F.d85493.2)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Derived.F.decl.loc10_50.1: @Derived.%Derived.F.type.loc10_50.1 (%Derived.F.type.77405e.1) = fn_decl @Derived.F.loc10_50.1 [symbolic = @Derived.%Derived.F.loc10_50.1 (constants.%Derived.F.d85493.1)] {
// CHECK:STDOUT:       %self.param_patt: @Derived.F.loc10_50.1.%pattern_type.loc10_28 (%pattern_type.839) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @Derived.F.loc10_50.1.%pattern_type.loc10_28 (%pattern_type.839) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @Derived.F.loc10_50.1.%pattern_type.loc10_44 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @Derived.F.loc10_50.1.%pattern_type.loc10_44 (%pattern_type.4f4) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @Derived.F.loc10_50.1.%Derived (%Derived) = value_param call_param0
// CHECK:STDOUT:       %.loc10_30.1: type = splice_block %Self.ref [symbolic = %Derived (constants.%Derived)] {
// CHECK:STDOUT:         %.loc10_30.2: type = specific_constant constants.%Derived, @Derived(constants.%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_30.2 [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @Derived.F.loc10_50.1.%Derived (%Derived) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @Derived.F.loc10_50.1.%ptr.loc10_47.1 (%ptr.e8f) = value_param call_param1
// CHECK:STDOUT:       %.loc10_47: type = splice_block %ptr.loc10_47.2 [symbolic = %ptr.loc10_47.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:         %T.ref: type = name_ref T, @Derived.%T.loc7_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:         %ptr.loc10_47.2: type = ptr_type %T.ref [symbolic = %ptr.loc10_47.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %t: @Derived.F.loc10_50.1.%ptr.loc10_47.1 (%ptr.e8f) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Derived.F.decl.loc10_50.2: @Derived.%Derived.F.type.loc10_50.2 (%Derived.F.type.77405e.2) = fn_decl @Derived.F.loc10_50.2 [symbolic = @Derived.%Derived.F.loc10_50.2 (constants.%Derived.F.d85493.2)] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc12_1.1
// CHECK:STDOUT:     vtable_decl = %vtable_decl
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Derived
// CHECK:STDOUT:     .Base = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     .F [hidden] = %Derived.F.decl.loc10_50.1
// CHECK:STDOUT:     extend %Base.loc8_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Base.vtable {
// CHECK:STDOUT:   @Base.%Base.F.specific_fn.loc6_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: vtable @Derived.vtable {
// CHECK:STDOUT:   @Derived.%Derived.F.specific_fn.loc12_1.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic override fn @Derived.F.loc10_50.1(@Derived.%T.loc7_16.2: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %pattern_type.loc10_28: type = pattern_type %Derived [symbolic = %pattern_type.loc10_28 (constants.%pattern_type.839)]
// CHECK:STDOUT:   %ptr.loc10_47.1: type = ptr_type %T [symbolic = %ptr.loc10_47.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type.loc10_44: type = pattern_type %ptr.loc10_47.1 [symbolic = %pattern_type.loc10_44 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_28: <witness> = require_complete_type %Derived [symbolic = %require_complete.loc10_28 (constants.%require_complete.1e7)]
// CHECK:STDOUT:   %require_complete.loc10_44: <witness> = require_complete_type %ptr.loc10_47.1 [symbolic = %require_complete.loc10_44 (constants.%require_complete.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   override fn(%self.param: @Derived.F.loc10_50.1.%Derived (%Derived), %t.param: @Derived.F.loc10_50.1.%ptr.loc10_47.1 (%ptr.e8f)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic override fn @Derived.F.loc10_50.2(@Derived.%T.loc7_16.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Derived.F.type: type = fn_type @Derived.F.loc10_50.1, @Derived(%T) [symbolic = %Derived.F.type (constants.%Derived.F.type.77405e.1)]
// CHECK:STDOUT:   %Derived.F: @Derived.F.loc10_50.2.%Derived.F.type (%Derived.F.type.77405e.1) = struct_value () [symbolic = %Derived.F (constants.%Derived.F.d85493.1)]
// CHECK:STDOUT:   %Derived: type = class_type @Derived, @Derived(%T) [symbolic = %Derived (constants.%Derived)]
// CHECK:STDOUT:   %Derived.F.specific_fn.loc10_50.2: <specific function> = specific_function %Derived.F, @Derived.F.loc10_50.1(%T) [symbolic = %Derived.F.specific_fn.loc10_50.2 (constants.%Derived.F.specific_fn.822c1a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   override fn(%self.param: @Derived.F.loc10_50.2.%Base (%Base.99a), %t.param: @Derived.F.loc10_50.2.%ptr (%ptr.e8f)) [thunk @Derived.%Derived.F.decl.loc10_50.1 for @Base.%Base.F.decl, @Base.F(constants.%ptr.e8f)] {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc10_50.1: @Derived.F.loc10_50.2.%Derived.F.type (%Derived.F.type.77405e.1) = specific_constant @Derived.%Derived.F.decl.loc10_50.1, @Derived(constants.%T) [symbolic = %Derived.F (constants.%Derived.F.d85493.1)]
// CHECK:STDOUT:     %F.ref: @Derived.F.loc10_50.2.%Derived.F.type (%Derived.F.type.77405e.1) = name_ref F, %.loc10_50.1 [symbolic = %Derived.F (constants.%Derived.F.d85493.1)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %.loc10_50.2: ref @Derived.F.loc10_50.2.%Derived (%Derived) = enclosing_class_access %.loc5, element0
// CHECK:STDOUT:     %.loc10_50.3: @Derived.F.loc10_50.2.%Derived (%Derived) = acquire_value %.loc10_50.2
// CHECK:STDOUT:     %Derived.F.bound: <bound method> = bound_method %.loc10_50.3, %F.ref
// CHECK:STDOUT:     %Derived.F.specific_fn.loc10_50.1: <specific function> = specific_function %F.ref, @Derived.F.loc10_50.1(constants.%T) [symbolic = %Derived.F.specific_fn.loc10_50.2 (constants.%Derived.F.specific_fn.822c1a.2)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %.loc10_50.3, %Derived.F.specific_fn.loc10_50.1
// CHECK:STDOUT:     %Derived.F.call: init %empty_tuple.type = call %bound_method(%.loc10_50.3, %t.param)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived(constants.%T) {
// CHECK:STDOUT:   %T.loc7_16.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ptr.loc8_22.2 => constants.%ptr.e8f
// CHECK:STDOUT:   %Base.loc8_23.2 => constants.%Base.99a
// CHECK:STDOUT:   %require_complete => constants.%require_complete.29f
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.elem => constants.%Derived.elem
// CHECK:STDOUT:   %Derived.F.type.loc10_50.1 => constants.%Derived.F.type.77405e.1
// CHECK:STDOUT:   %Derived.F.loc10_50.1 => constants.%Derived.F.d85493.1
// CHECK:STDOUT:   %Derived.F.type.loc10_50.2 => constants.%Derived.F.type.77405e.2
// CHECK:STDOUT:   %Derived.F.loc10_50.2 => constants.%Derived.F.d85493.2
// CHECK:STDOUT:   %Derived.F.specific_fn.loc12_1.2 => constants.%Derived.F.specific_fn.822c1a.1
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.619
// CHECK:STDOUT:   %complete_type.loc12_1.2 => constants.%complete_type.db5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.F.loc10_50.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %pattern_type.loc10_28 => constants.%pattern_type.839
// CHECK:STDOUT:   %ptr.loc10_47.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type.loc10_44 => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_28 => constants.%require_complete.1e7
// CHECK:STDOUT:   %require_complete.loc10_44 => constants.%require_complete.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Derived.F.loc10_50.2(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ptr => constants.%ptr.e8f
// CHECK:STDOUT:   %Base => constants.%Base.99a
// CHECK:STDOUT:   %pattern_type.loc5_27 => constants.%pattern_type.bdc
// CHECK:STDOUT:   %pattern_type.loc5_43 => constants.%pattern_type.4f4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc5_27 => constants.%require_complete.29f
// CHECK:STDOUT:   %require_complete.loc5_43 => constants.%require_complete.ef1
// CHECK:STDOUT:   %Derived.F.type => constants.%Derived.F.type.77405e.1
// CHECK:STDOUT:   %Derived.F => constants.%Derived.F.d85493.1
// CHECK:STDOUT:   %Derived => constants.%Derived
// CHECK:STDOUT:   %Derived.F.specific_fn.loc10_50.2 => constants.%Derived.F.specific_fn.822c1a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
