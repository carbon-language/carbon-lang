// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
// INCLUDE-FILE: toolchain/testing/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/min_prelude/virtual_modifiers.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/min_prelude/virtual_modifiers.carbon

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

class Derived {
  extend base: Modifiers.Base;
  impl fn H[self: Self]();
}

fn Use() {
  //@dump-sem-ir-begin
  var d: Derived = {.base = {}};
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
  var v: Modifiers.Base = {};
  //@dump-sem-ir-end
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
  //@dump-sem-ir-begin
  var b1: B1 = {};
  var b2: B2 = {.base = {}};
  var c: C = {.base = {.base = {}}};
  //@dump-sem-ir-end
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
  var m1: ();
  var m2: ();

  virtual fn F[self: Self]();
}

fn F() {
  var i: () = ();
  //@dump-sem-ir-begin
  // TODO: These should initialize element1 (.m), not element0 (the vptr)
  var b1: Base = {.m2 = i, .m1 = i};
  var b2: Base = {.m2 = (), .m1 = ()};

  // This one is good, though.
  b1.m2 = ();
  //@dump-sem-ir-end
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
  // CHECK:STDERR:   impl fn F[self: Self](v: ());
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: fail_impl_mismatch.carbon:[[@LINE-8]]:3: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self]();
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F[self: Self](v: ());
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

// --- fail_todo_impl_generic_base.carbon

library "[[@TEST_NAME]]";

class T1 {
}

base class Base(T:! type) {
  virtual fn F[self: Self](t: T);
}

class Derived {
  extend base: Base(T1);
  // CHECK:STDERR: fail_todo_impl_generic_base.carbon:[[@LINE+7]]:25: error: type `<pattern for T1>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for T>` [RedeclParamDiffersType]
  // CHECK:STDERR:   impl fn F[self: Self](t: T1);
  // CHECK:STDERR:                         ^~~~~
  // CHECK:STDERR: fail_todo_impl_generic_base.carbon:[[@LINE-8]]:28: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
  // CHECK:STDERR:   virtual fn F[self: Self](t: T);
  // CHECK:STDERR:                            ^~~~
  // CHECK:STDERR:
  impl fn F[self: Self](t: T1);
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
  virtual fn F[self: Self]();
}

// --- with_dependent_arg.carbon

library "[[@TEST_NAME]]";

base class T1(T:! type) {
  virtual fn F[self: Self](t: T);
}


// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.c3d: <vtable> = vtable (%H) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %pattern_type.41e: type = pattern_type %Abstract [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %.2b2: <vtable> = vtable (%J, %K) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
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
// CHECK:STDOUT:   %.loc7: <vtable> = vtable (%H.decl) [concrete = constants.%.c3d]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %.loc13: <vtable> = vtable (%J.decl, %K.decl) [concrete = constants.%.2b2]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete = constants.%struct_type.vptr]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
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
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = binding_pattern d [concrete]
// CHECK:STDOUT:     %d.var_patt: %pattern_type.fb9 = var_pattern %d.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %Derived = var %d.var_patt
// CHECK:STDOUT:   %.loc13_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc13_31.1: %struct_type.base.f5e = struct_literal (%.loc13_30.1)
// CHECK:STDOUT:   %.loc13_31.2: ref %Base = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc13_30.2: ref %ptr.454 = class_element_access %.loc13_31.2, element0
// CHECK:STDOUT:   %.loc13_30.3: ref %ptr.454 = vtable_ptr @Derived.%.loc9
// CHECK:STDOUT:   %.loc13_30.4: init %ptr.454 = initialize_from %.loc13_30.3 to %.loc13_30.2
// CHECK:STDOUT:   %.loc13_30.5: init %Base = class_init (%.loc13_30.4), %.loc13_31.2
// CHECK:STDOUT:   %.loc13_31.3: init %Base = converted %.loc13_30.1, %.loc13_30.5
// CHECK:STDOUT:   %.loc13_31.4: init %Derived = class_init (%.loc13_31.3), %d.var
// CHECK:STDOUT:   %.loc13_3: init %Derived = converted %.loc13_31.1, %.loc13_31.4
// CHECK:STDOUT:   assign %d.var, %.loc13_3
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d: ref %Derived = bind_name d, %d.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [concrete] {
// CHECK:STDOUT:     .Base = %Modifiers.Base
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.Base: type = import_ref Modifiers//default, Base, loaded [concrete = constants.%Base]
// CHECK:STDOUT:   %Modifiers.import_ref.ace = import_ref Modifiers//default, loc7_1, unloaded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base = var %v.var_patt
// CHECK:STDOUT:   %.loc8_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc8_28.2: ref %ptr.454 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc8_28.3: ref %ptr.454 = vtable_ptr imports.%Modifiers.import_ref.ace
// CHECK:STDOUT:   %.loc8_28.4: init %ptr.454 = initialize_from %.loc8_28.3 to %.loc8_28.2
// CHECK:STDOUT:   %.loc8_28.5: init %Base = class_init (%.loc8_28.4), %v.var
// CHECK:STDOUT:   %.loc8_3: init %Base = converted %.loc8_28.1, %.loc8_28.5
// CHECK:STDOUT:   assign %v.var, %.loc8_3
// CHECK:STDOUT:   %.loc8_19: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %pattern_type.e47: type = pattern_type %B1 [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %pattern_type.b19: type = pattern_type %B2 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.base.a0c: type = struct_type {.base: %struct_type.base.f5e} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.e47 = binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.e47 = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %B1 = var %b1.var_patt
// CHECK:STDOUT:   %.loc20_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc20_17.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc20_17.3: ref %ptr.454 = vtable_ptr @B1.%.loc6
// CHECK:STDOUT:   %.loc20_17.4: init %ptr.454 = initialize_from %.loc20_17.3 to %.loc20_17.2
// CHECK:STDOUT:   %.loc20_17.5: init %B1 = class_init (%.loc20_17.4), %b1.var
// CHECK:STDOUT:   %.loc20_3: init %B1 = converted %.loc20_17.1, %.loc20_17.5
// CHECK:STDOUT:   assign %b1.var, %.loc20_3
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %b1: ref %B1 = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.b19 = binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.b19 = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %B2 = var %b2.var_patt
// CHECK:STDOUT:   %.loc21_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc21_27.1: %struct_type.base.f5e = struct_literal (%.loc21_26.1)
// CHECK:STDOUT:   %.loc21_27.2: ref %B1 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc21_26.2: ref %ptr.454 = class_element_access %.loc21_27.2, element0
// CHECK:STDOUT:   %.loc21_26.3: ref %ptr.454 = vtable_ptr @B2.%.loc11
// CHECK:STDOUT:   %.loc21_26.4: init %ptr.454 = initialize_from %.loc21_26.3 to %.loc21_26.2
// CHECK:STDOUT:   %.loc21_26.5: init %B1 = class_init (%.loc21_26.4), %.loc21_27.2
// CHECK:STDOUT:   %.loc21_27.3: init %B1 = converted %.loc21_26.1, %.loc21_26.5
// CHECK:STDOUT:   %.loc21_27.4: init %B2 = class_init (%.loc21_27.3), %b2.var
// CHECK:STDOUT:   %.loc21_3: init %B2 = converted %.loc21_27.1, %.loc21_27.4
// CHECK:STDOUT:   assign %b2.var, %.loc21_3
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %b2: ref %B2 = bind_name b2, %b2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:   %.loc22_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc22_34.1: %struct_type.base.f5e = struct_literal (%.loc22_33.1)
// CHECK:STDOUT:   %.loc22_35.1: %struct_type.base.a0c = struct_literal (%.loc22_34.1)
// CHECK:STDOUT:   %.loc22_35.2: ref %B2 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc22_34.2: ref %B1 = class_element_access %.loc22_35.2, element0
// CHECK:STDOUT:   %.loc22_33.2: ref %ptr.454 = class_element_access %.loc22_34.2, element0
// CHECK:STDOUT:   %.loc22_33.3: ref %ptr.454 = vtable_ptr @C.%.loc16
// CHECK:STDOUT:   %.loc22_33.4: init %ptr.454 = initialize_from %.loc22_33.3 to %.loc22_33.2
// CHECK:STDOUT:   %.loc22_33.5: init %B1 = class_init (%.loc22_33.4), %.loc22_34.2
// CHECK:STDOUT:   %.loc22_34.3: init %B1 = converted %.loc22_33.1, %.loc22_33.5
// CHECK:STDOUT:   %.loc22_34.4: init %B2 = class_init (%.loc22_34.3), %.loc22_35.2
// CHECK:STDOUT:   %.loc22_35.3: init %B2 = converted %.loc22_34.1, %.loc22_34.4
// CHECK:STDOUT:   %.loc22_35.4: init %C = class_init (%.loc22_35.3), %c.var
// CHECK:STDOUT:   %.loc22_3: init %C = converted %.loc22_35.1, %.loc22_35.4
// CHECK:STDOUT:   assign %c.var, %.loc22_3
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %pattern_type.bcc: type = pattern_type %Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %struct_type.m2.m1: type = struct_type {.m2: %empty_tuple.type, .m1: %empty_tuple.type} [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %pattern_type.bcc = binding_pattern b1 [concrete]
// CHECK:STDOUT:     %b1.var_patt: %pattern_type.bcc = var_pattern %b1.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %Base = var %b1.var_patt
// CHECK:STDOUT:   %i.ref.loc15_25: ref %empty_tuple.type = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc15_34: ref %empty_tuple.type = name_ref i, %i
// CHECK:STDOUT:   %.loc15_35.1: %struct_type.m2.m1 = struct_literal (%i.ref.loc15_25, %i.ref.loc15_34)
// CHECK:STDOUT:   %.loc15_35.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc15_35.3: ref %ptr.454 = vtable_ptr @Base.%.loc9
// CHECK:STDOUT:   %.loc15_35.4: init %ptr.454 = initialize_from %.loc15_35.3 to %.loc15_35.2
// CHECK:STDOUT:   %.loc15_35.5: ref %empty_tuple.type = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc15_34: init %empty_tuple.type = tuple_init () to %.loc15_35.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_35.6: init %empty_tuple.type = converted %i.ref.loc15_34, %.loc15_34 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_35.7: ref %empty_tuple.type = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc15_25: init %empty_tuple.type = tuple_init () to %.loc15_35.7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_35.8: init %empty_tuple.type = converted %i.ref.loc15_25, %.loc15_25 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc15_35.9: init %Base = class_init (%.loc15_35.4, %.loc15_35.6, %.loc15_35.8), %b1.var
// CHECK:STDOUT:   %.loc15_3: init %Base = converted %.loc15_35.1, %.loc15_35.9
// CHECK:STDOUT:   assign %b1.var, %.loc15_3
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %pattern_type.bcc = binding_pattern b2 [concrete]
// CHECK:STDOUT:     %b2.var_patt: %pattern_type.bcc = var_pattern %b2.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var %b2.var_patt
// CHECK:STDOUT:   %.loc16_26.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_36.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc16_37.1: %struct_type.m2.m1 = struct_literal (%.loc16_26.1, %.loc16_36.1)
// CHECK:STDOUT:   %.loc16_37.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc16_37.3: ref %ptr.454 = vtable_ptr @Base.%.loc9
// CHECK:STDOUT:   %.loc16_37.4: init %ptr.454 = initialize_from %.loc16_37.3 to %.loc16_37.2
// CHECK:STDOUT:   %.loc16_37.5: ref %empty_tuple.type = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc16_36.2: init %empty_tuple.type = tuple_init () to %.loc16_37.5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_37.6: init %empty_tuple.type = converted %.loc16_36.1, %.loc16_36.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_37.7: ref %empty_tuple.type = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc16_26.2: init %empty_tuple.type = tuple_init () to %.loc16_37.7 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_37.8: init %empty_tuple.type = converted %.loc16_26.1, %.loc16_26.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc16_37.9: init %Base = class_init (%.loc16_37.4, %.loc16_37.6, %.loc16_37.8), %b2.var
// CHECK:STDOUT:   %.loc16_3: init %Base = converted %.loc16_37.1, %.loc16_37.9
// CHECK:STDOUT:   assign %b2.var, %.loc16_3
// CHECK:STDOUT:   %Base.ref.loc16: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = bind_name b2, %b2.var
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %Base.elem = name_ref m2, @Base.%.loc6_9 [concrete = @Base.%.loc6_9]
// CHECK:STDOUT:   %.loc19_5: ref %empty_tuple.type = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %.loc19_12.1: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:   %.loc19_12.2: init %empty_tuple.type = tuple_init () to %.loc19_5 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc19_9: init %empty_tuple.type = converted %.loc19_12.1, %.loc19_12.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   assign %.loc19_5, %.loc19_9
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
