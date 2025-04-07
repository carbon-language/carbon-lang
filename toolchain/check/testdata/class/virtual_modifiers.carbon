// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

// --- fail_todo_impl_generic_base.carbon

library "[[@TEST_NAME]]";

class T1 {
}

base class Base(T:! type) {
  virtual fn F[self: Self](t: T);
}

class Derived {
  extend base: Base(T1);
  // CHECK:STDERR: fail_todo_impl_generic_base.carbon:[[@LINE+7]]:25: error: type `T1` of parameter 1 in redeclaration differs from previous parameter type `T` [RedeclParamDiffersType]
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

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.c3d: <vtable> = vtable (%H) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [concrete]
// CHECK:STDOUT:   %J.type: type = fn_type @J [concrete]
// CHECK:STDOUT:   %J: %J.type = struct_value () [concrete]
// CHECK:STDOUT:   %K.type: type = fn_type @K [concrete]
// CHECK:STDOUT:   %K: %K.type = struct_value () [concrete]
// CHECK:STDOUT:   %.2b2: <vtable> = vtable (%J, %K) [concrete]
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
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%H.decl) [concrete = constants.%.c3d]
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
// CHECK:STDOUT:     %self.patt: %Abstract = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Abstract = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [concrete = constants.%K] {
// CHECK:STDOUT:     %self.patt: %Abstract = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Abstract = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Abstract = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Abstract [concrete = constants.%Abstract]
// CHECK:STDOUT:     %self: %Abstract = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12: <vtable> = vtable (%J.decl, %K.decl) [concrete = constants.%.2b2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @H[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @J[%self.param_patt: %Abstract]();
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @K[%self.param_patt: %Abstract]();
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
// CHECK:STDOUT:   %H.type.dba: type = fn_type @H.1 [concrete]
// CHECK:STDOUT:   %H.bce: %H.type.dba = struct_value () [concrete]
// CHECK:STDOUT:   %.dce: <vtable> = vtable (%H.bce) [concrete]
// CHECK:STDOUT:   %struct_type.base.96c: type = struct_type {.base: %Base} [concrete]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base.96c [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
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
// CHECK:STDOUT:   %H.decl: %H.type.dba = fn_decl @H.1 [concrete = constants.%H.bce] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (%H.decl) [concrete = constants.%.dce]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.96c [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @H.1[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2[%self.param_patt: %Base]() [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %.loc12_3.1: %Derived = var_pattern %d.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %d.var: ref %Derived = var d
// CHECK:STDOUT:   %.loc12_30.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_31.1: %struct_type.base.f5e = struct_literal (%.loc12_30.1)
// CHECK:STDOUT:   %.loc12_31.2: ref %Base = class_element_access %d.var, element0
// CHECK:STDOUT:   %.loc12_30.2: ref %ptr.454 = class_element_access %.loc12_31.2, element0
// CHECK:STDOUT:   %.loc12_30.3: ref %ptr.454 = vtable_ptr @Derived.%.loc9
// CHECK:STDOUT:   %.loc12_30.4: init %ptr.454 = initialize_from %.loc12_30.3 to %.loc12_30.2
// CHECK:STDOUT:   %.loc12_30.5: init %Base = class_init (%.loc12_30.4), %.loc12_31.2
// CHECK:STDOUT:   %.loc12_31.3: init %Base = converted %.loc12_30.1, %.loc12_30.5
// CHECK:STDOUT:   %.loc12_31.4: init %Derived = class_init (%.loc12_31.3), %d.var
// CHECK:STDOUT:   %.loc12_3.2: init %Derived = converted %.loc12_31.1, %.loc12_31.4
// CHECK:STDOUT:   assign %d.var, %.loc12_3.2
// CHECK:STDOUT:   %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:   %d: ref %Derived = bind_name d, %d.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %.02f: <vtable> = vtable (%H, %F) [concrete]
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
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (constants.%H, %F.decl) [concrete = constants.%.02f]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H[%self.param_patt: %Base]() [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
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
// CHECK:STDOUT:   %Modifiers.import_ref.ace = import_ref Modifiers//default, loc6_1, unloaded
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Modifiers.import_ref.05e
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Modifiers.import_ref.1f3
// CHECK:STDOUT:   .H = imports.%Modifiers.import_ref.2cc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %Base = binding_pattern v
// CHECK:STDOUT:     %.loc7_3.1: %Base = var_pattern %v.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Base = var v
// CHECK:STDOUT:   %.loc7_28.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc7_28.2: ref %ptr.454 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc7_28.3: ref %ptr.454 = vtable_ptr imports.%Modifiers.import_ref.ace
// CHECK:STDOUT:   %.loc7_28.4: init %ptr.454 = initialize_from %.loc7_28.3 to %.loc7_28.2
// CHECK:STDOUT:   %.loc7_28.5: init %Base = class_init (%.loc7_28.4), %v.var
// CHECK:STDOUT:   %.loc7_3.2: init %Base = converted %.loc7_28.1, %.loc7_28.5
// CHECK:STDOUT:   assign %v.var, %.loc7_3.2
// CHECK:STDOUT:   %.loc7_19: type = splice_block %Base.ref [concrete = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [concrete = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%Modifiers.Base [concrete = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [concrete]
// CHECK:STDOUT:   %F.type.13a: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.df5: %F.type.13a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.593: <vtable> = vtable (%F.df5) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %A2: type = class_type @A2 [concrete]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %A1 [concrete]
// CHECK:STDOUT:   %F.type.4ae: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.1d5: %F.type.4ae = struct_value () [concrete]
// CHECK:STDOUT:   %.943: <vtable> = vtable (%F.1d5) [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.13a = fn_decl @F.1 [concrete = constants.%F.df5] {
// CHECK:STDOUT:     %self.patt: %A1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %A1 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A1 [concrete = constants.%A1]
// CHECK:STDOUT:     %self: %A1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [concrete = constants.%.593]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %A1.ref: type = name_ref A1, file.%A1.decl [concrete = constants.%A1]
// CHECK:STDOUT:   %.loc9: %A2.elem = base_decl %A1.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.4ae = fn_decl @F.2 [concrete = constants.%F.1d5] {
// CHECK:STDOUT:     %self.patt: %A2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %A2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %A2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%A2 [concrete = constants.%A2]
// CHECK:STDOUT:     %self: %A2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <vtable> = vtable (%F.decl) [concrete = constants.%.943]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.a6f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .A1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %A1]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %A2]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = class_type @B1 [concrete]
// CHECK:STDOUT:   %F.type.e4c: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.8f5: %F.type.e4c = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.bc5: <vtable> = vtable (%F.8f5) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %B2: type = class_type @B2 [concrete]
// CHECK:STDOUT:   %B2.elem: type = unbound_element_type %B2, %B1 [concrete]
// CHECK:STDOUT:   %F.type.b26: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.d48: %F.type.b26 = struct_value () [concrete]
// CHECK:STDOUT:   %.579: <vtable> = vtable (%F.d48) [concrete]
// CHECK:STDOUT:   %struct_type.base.508: type = struct_type {.base: %B1} [concrete]
// CHECK:STDOUT:   %complete_type.5ac: <witness> = complete_type_witness %struct_type.base.508 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B2 [concrete]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.3 [concrete]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [concrete]
// CHECK:STDOUT:   %.5f6: <vtable> = vtable (%F.437) [concrete]
// CHECK:STDOUT:   %struct_type.base.421: type = struct_type {.base: %B2} [concrete]
// CHECK:STDOUT:   %complete_type.066: <witness> = complete_type_witness %struct_type.base.421 [concrete]
// CHECK:STDOUT:   %Use.type: type = fn_type @Use [concrete]
// CHECK:STDOUT:   %Use: %Use.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %struct_type.base.f5e: type = struct_type {.base: %empty_struct_type} [concrete]
// CHECK:STDOUT:   %struct_type.base.a0c: type = struct_type {.base: %struct_type.base.f5e} [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.e4c = fn_decl @F.1 [concrete = constants.%F.8f5] {
// CHECK:STDOUT:     %self.patt: %B1 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %B1 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B1 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B1 [concrete = constants.%B1]
// CHECK:STDOUT:     %self: %B1 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [concrete = constants.%.bc5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %.loc9: %B2.elem = base_decl %B1.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.b26 = fn_decl @F.2 [concrete = constants.%F.d48] {
// CHECK:STDOUT:     %self.patt: %B2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %B2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %B2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%B2 [concrete = constants.%B2]
// CHECK:STDOUT:     %self: %B2 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc11: <vtable> = vtable (%F.decl) [concrete = constants.%.579]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.508 [concrete = constants.%complete_type.5ac]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.decl: %F.type.c29 = fn_decl @F.3 [concrete = constants.%F.437] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16: <vtable> = vtable (%F.decl) [concrete = constants.%.5f6]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.421 [concrete = constants.%complete_type.066]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .B2 = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B2.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %B1]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %B2]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.3[%self.param_patt: %C]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Use() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %B1 = binding_pattern b1
// CHECK:STDOUT:     %.loc19_3.1: %B1 = var_pattern %b1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %B1 = var b1
// CHECK:STDOUT:   %.loc19_17.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc19_17.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc19_17.3: ref %ptr.454 = vtable_ptr @B1.%.loc6
// CHECK:STDOUT:   %.loc19_17.4: init %ptr.454 = initialize_from %.loc19_17.3 to %.loc19_17.2
// CHECK:STDOUT:   %.loc19_17.5: init %B1 = class_init (%.loc19_17.4), %b1.var
// CHECK:STDOUT:   %.loc19_3.2: init %B1 = converted %.loc19_17.1, %.loc19_17.5
// CHECK:STDOUT:   assign %b1.var, %.loc19_3.2
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [concrete = constants.%B1]
// CHECK:STDOUT:   %b1: ref %B1 = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %B2 = binding_pattern b2
// CHECK:STDOUT:     %.loc20_3.1: %B2 = var_pattern %b2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %B2 = var b2
// CHECK:STDOUT:   %.loc20_26.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc20_27.1: %struct_type.base.f5e = struct_literal (%.loc20_26.1)
// CHECK:STDOUT:   %.loc20_27.2: ref %B1 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc20_26.2: ref %ptr.454 = class_element_access %.loc20_27.2, element0
// CHECK:STDOUT:   %.loc20_26.3: ref %ptr.454 = vtable_ptr @B2.%.loc11
// CHECK:STDOUT:   %.loc20_26.4: init %ptr.454 = initialize_from %.loc20_26.3 to %.loc20_26.2
// CHECK:STDOUT:   %.loc20_26.5: init %B1 = class_init (%.loc20_26.4), %.loc20_27.2
// CHECK:STDOUT:   %.loc20_27.3: init %B1 = converted %.loc20_26.1, %.loc20_26.5
// CHECK:STDOUT:   %.loc20_27.4: init %B2 = class_init (%.loc20_27.3), %b2.var
// CHECK:STDOUT:   %.loc20_3.2: init %B2 = converted %.loc20_27.1, %.loc20_27.4
// CHECK:STDOUT:   assign %b2.var, %.loc20_3.2
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [concrete = constants.%B2]
// CHECK:STDOUT:   %b2: ref %B2 = bind_name b2, %b2.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %.loc21_3.1: %C = var_pattern %c.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %c.var: ref %C = var c
// CHECK:STDOUT:   %.loc21_33.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc21_34.1: %struct_type.base.f5e = struct_literal (%.loc21_33.1)
// CHECK:STDOUT:   %.loc21_35.1: %struct_type.base.a0c = struct_literal (%.loc21_34.1)
// CHECK:STDOUT:   %.loc21_35.2: ref %B2 = class_element_access %c.var, element0
// CHECK:STDOUT:   %.loc21_34.2: ref %B1 = class_element_access %.loc21_35.2, element0
// CHECK:STDOUT:   %.loc21_33.2: ref %ptr.454 = class_element_access %.loc21_34.2, element0
// CHECK:STDOUT:   %.loc21_33.3: ref %ptr.454 = vtable_ptr @C.%.loc16
// CHECK:STDOUT:   %.loc21_33.4: init %ptr.454 = initialize_from %.loc21_33.3 to %.loc21_33.2
// CHECK:STDOUT:   %.loc21_33.5: init %B1 = class_init (%.loc21_33.4), %.loc21_34.2
// CHECK:STDOUT:   %.loc21_34.3: init %B1 = converted %.loc21_33.1, %.loc21_33.5
// CHECK:STDOUT:   %.loc21_34.4: init %B2 = class_init (%.loc21_34.3), %.loc21_35.2
// CHECK:STDOUT:   %.loc21_35.3: init %B2 = converted %.loc21_34.1, %.loc21_34.4
// CHECK:STDOUT:   %.loc21_35.4: init %C = class_init (%.loc21_35.3), %c.var
// CHECK:STDOUT:   %.loc21_3.2: init %C = converted %.loc21_35.1, %.loc21_35.4
// CHECK:STDOUT:   assign %c.var, %.loc21_3.2
// CHECK:STDOUT:   %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.f2b: <vtable> = vtable () [concrete]
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
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <vtable> = vtable () [concrete = constants.%.f2b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F[%self.param_patt: %C]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.5ee: <vtable> = vtable (%F.d17) [concrete]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [concrete]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
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
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [concrete]
// CHECK:STDOUT:   %bound_method.1da: <bound method> = bound_method %int_4.0c1, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5_9: %Base.elem = field_decl m1, element1 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %Base.elem = var_pattern %.loc5_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %Base.elem = var <none>
// CHECK:STDOUT:   %.loc6_9: %Base.elem = field_decl m2, element2 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %Base.elem = var_pattern %.loc6_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %Base.elem = var <none>
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.1 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (%F.decl) [concrete = constants.%.5ee]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [concrete = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .m1 = %.loc5_9
// CHECK:STDOUT:   .m2 = %.loc6_9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %i32 = binding_pattern i
// CHECK:STDOUT:     %.loc12_3.1: %i32 = var_pattern %i.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %i32 = var i
// CHECK:STDOUT:   %int_3.loc12: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc12: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc12_3.1: <bound method> = bound_method %int_3.loc12, %impl.elem0.loc12 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc12: <specific function> = specific_function %impl.elem0.loc12, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc12_3.2: <bound method> = bound_method %int_3.loc12, %specific_fn.loc12 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %bound_method.loc12_3.2(%int_3.loc12) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_3.2: init %i32 = converted %int_3.loc12, %int.convert_checked.loc12 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   assign %i.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i: ref %i32 = bind_name i, %i.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b1.patt: %Base = binding_pattern b1
// CHECK:STDOUT:     %.loc14_3.1: %Base = var_pattern %b1.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b1.var: ref %Base = var b1
// CHECK:STDOUT:   %i.ref.loc14_25: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc14_34: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc14_35.1: %struct_type.m2.m1.68c = struct_literal (%i.ref.loc14_25, %i.ref.loc14_34)
// CHECK:STDOUT:   %.loc14_35.2: ref %ptr.454 = class_element_access %b1.var, element0
// CHECK:STDOUT:   %.loc14_35.3: ref %ptr.454 = vtable_ptr @Base.%.loc9
// CHECK:STDOUT:   %.loc14_35.4: init %ptr.454 = initialize_from %.loc14_35.3 to %.loc14_35.2
// CHECK:STDOUT:   %.loc14_34: %i32 = bind_value %i.ref.loc14_34
// CHECK:STDOUT:   %.loc14_35.5: ref %i32 = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc14_35.6: init %i32 = initialize_from %.loc14_34 to %.loc14_35.5
// CHECK:STDOUT:   %.loc14_25: %i32 = bind_value %i.ref.loc14_25
// CHECK:STDOUT:   %.loc14_35.7: ref %i32 = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc14_35.8: init %i32 = initialize_from %.loc14_25 to %.loc14_35.7
// CHECK:STDOUT:   %.loc14_35.9: init %Base = class_init (%.loc14_35.4, %.loc14_35.6, %.loc14_35.8), %b1.var
// CHECK:STDOUT:   %.loc14_3.2: init %Base = converted %.loc14_35.1, %.loc14_35.9
// CHECK:STDOUT:   assign %b1.var, %.loc14_3.2
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %Base = binding_pattern b2
// CHECK:STDOUT:     %.loc15_3.1: %Base = var_pattern %b2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var b2
// CHECK:STDOUT:   %int_3.loc15: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [concrete = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc15_35.1: %struct_type.m2.m1.5f2 = struct_literal (%int_3.loc15, %int_5)
// CHECK:STDOUT:   %.loc15_35.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc15_35.3: ref %ptr.454 = vtable_ptr @Base.%.loc9
// CHECK:STDOUT:   %.loc15_35.4: init %ptr.454 = initialize_from %.loc15_35.3 to %.loc15_35.2
// CHECK:STDOUT:   %impl.elem0.loc15_35.1: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_35.1: <bound method> = bound_method %int_5, %impl.elem0.loc15_35.1 [concrete = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %specific_fn.loc15_35.1: <specific function> = specific_function %impl.elem0.loc15_35.1, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_35.2: <bound method> = bound_method %int_5, %specific_fn.loc15_35.1 [concrete = constants.%bound_method.a25]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.1: init %i32 = call %bound_method.loc15_35.2(%int_5) [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.5: init %i32 = converted %int_5, %int.convert_checked.loc15_35.1 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.6: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc15_35.7: init %i32 = initialize_from %.loc15_35.5 to %.loc15_35.6 [concrete = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc15_35.2: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_35.3: <bound method> = bound_method %int_3.loc15, %impl.elem0.loc15_35.2 [concrete = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %specific_fn.loc15_35.2: <specific function> = specific_function %impl.elem0.loc15_35.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_35.4: <bound method> = bound_method %int_3.loc15, %specific_fn.loc15_35.2 [concrete = constants.%bound_method.047]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.2: init %i32 = call %bound_method.loc15_35.4(%int_3.loc15) [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.8: init %i32 = converted %int_3.loc15, %int.convert_checked.loc15_35.2 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.9: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc15_35.10: init %i32 = initialize_from %.loc15_35.8 to %.loc15_35.9 [concrete = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.11: init %Base = class_init (%.loc15_35.4, %.loc15_35.7, %.loc15_35.10), %b2.var
// CHECK:STDOUT:   %.loc15_3.2: init %Base = converted %.loc15_35.1, %.loc15_35.11
// CHECK:STDOUT:   assign %b2.var, %.loc15_3.2
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = bind_name b2, %b2.var
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %Base.elem = name_ref m2, @Base.%.loc6_9 [concrete = @Base.%.loc6_9]
// CHECK:STDOUT:   %.loc18_5: ref %i32 = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc18: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc18_9.1: <bound method> = bound_method %int_4, %impl.elem0.loc18 [concrete = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %specific_fn.loc18: <specific function> = specific_function %impl.elem0.loc18, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc18_9.2: <bound method> = bound_method %int_4, %specific_fn.loc18 [concrete = constants.%bound_method.1da]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %bound_method.loc18_9.2(%int_4) [concrete = constants.%int_4.940]
// CHECK:STDOUT:   %.loc18_9: init %i32 = converted %int_4, %int.convert_checked.loc18 [concrete = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc18_5, %.loc18_9
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
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.f2b: <vtable> = vtable () [concrete]
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
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc10: <vtable> = vtable () [concrete = constants.%.f2b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.base [concrete = constants.%complete_type.336]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractBase: type = class_type @AbstractBase [concrete]
// CHECK:STDOUT:   %F.type.85b: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.6e9: %F.type.85b = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.6ec: <vtable> = vtable (%F.6e9) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %AbstractIntermediate: type = class_type @AbstractIntermediate [concrete]
// CHECK:STDOUT:   %AbstractIntermediate.elem: type = unbound_element_type %AbstractIntermediate, %AbstractBase [concrete]
// CHECK:STDOUT:   %struct_type.base.efd: type = struct_type {.base: %AbstractBase} [concrete]
// CHECK:STDOUT:   %complete_type.2d3: <witness> = complete_type_witness %struct_type.base.efd [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %AbstractIntermediate [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.85b = fn_decl @F.1 [concrete = constants.%F.6e9] {
// CHECK:STDOUT:     %self.patt: %AbstractBase = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %AbstractBase = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %AbstractBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%AbstractBase [concrete = constants.%AbstractBase]
// CHECK:STDOUT:     %self: %AbstractBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [concrete = constants.%.6ec]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%AbstractBase
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractIntermediate {
// CHECK:STDOUT:   %AbstractBase.ref: type = name_ref AbstractBase, file.%AbstractBase.decl [concrete = constants.%AbstractBase]
// CHECK:STDOUT:   %.loc9: %AbstractIntermediate.elem = base_decl %AbstractBase.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc10: <vtable> = vtable (constants.%F.6e9) [concrete = constants.%.6ec]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.efd [concrete = constants.%complete_type.2d3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <vtable> = vtable (%F.decl) [concrete = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.da5 [concrete = constants.%complete_type.f8c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .AbstractIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %AbstractIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @F.1[%self.param_patt: %AbstractBase]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- virtual_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %VirtualBase: type = class_type @VirtualBase [concrete]
// CHECK:STDOUT:   %F.type.e62: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.3e7: %F.type.e62 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.def: <vtable> = vtable (%F.3e7) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %VirtualIntermediate: type = class_type @VirtualIntermediate [concrete]
// CHECK:STDOUT:   %VirtualIntermediate.elem: type = unbound_element_type %VirtualIntermediate, %VirtualBase [concrete]
// CHECK:STDOUT:   %struct_type.base.61e: type = struct_type {.base: %VirtualBase} [concrete]
// CHECK:STDOUT:   %complete_type.f09: <witness> = complete_type_witness %struct_type.base.61e [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %VirtualIntermediate [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.e62 = fn_decl @F.1 [concrete = constants.%F.3e7] {
// CHECK:STDOUT:     %self.patt: %VirtualBase = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %VirtualBase = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %VirtualBase = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%VirtualBase [concrete = constants.%VirtualBase]
// CHECK:STDOUT:     %self: %VirtualBase = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [concrete = constants.%.def]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%VirtualBase
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualIntermediate {
// CHECK:STDOUT:   %VirtualBase.ref: type = name_ref VirtualBase, file.%VirtualBase.decl [concrete = constants.%VirtualBase]
// CHECK:STDOUT:   %.loc9: %VirtualIntermediate.elem = base_decl %VirtualBase.ref, element0 [concrete]
// CHECK:STDOUT:   %.loc10: <vtable> = vtable (constants.%F.3e7) [concrete = constants.%.def]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.61e [concrete = constants.%complete_type.f09]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc15: <vtable> = vtable (%F.decl) [concrete = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.43c [concrete = constants.%complete_type.fa6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .VirtualIntermediate = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %VirtualIntermediate.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %VirtualBase]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %Derived]();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_impl_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.5ee: <vtable> = vtable (%F.d17) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.1 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [concrete = constants.%.5ee]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [concrete = constants.%Base]
// CHECK:STDOUT:   %.loc9: %Derived.elem = base_decl %Base.ref, element0 [concrete]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %v.patt: %i32 = binding_pattern v
// CHECK:STDOUT:     %v.param_patt: %i32 = value_param_pattern %v.patt, call_param1
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
// CHECK:STDOUT:   %.loc18: <vtable> = vtable (%F.decl) [concrete = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %Base]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %Derived](%v.param_patt: %i32);
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
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete]
// CHECK:STDOUT:   %Convert.type.c41: type = fn_type @Convert.2 [concrete]
// CHECK:STDOUT:   %Convert.f35: %Convert.type.c41 = struct_value () [concrete]
// CHECK:STDOUT:   %T1.val: %T1 = struct_value () [concrete]
// CHECK:STDOUT:   %Base: type = class_type @Base [concrete]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.5ee: <vtable> = vtable (%F.d17) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
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
// CHECK:STDOUT:   impl_decl @impl [concrete] {} {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.cc7 = name_ref ImplicitAs, imports.%Core.ImplicitAs [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%T1)> [concrete = constants.%ImplicitAs.type.e40]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%Convert.decl) [concrete = constants.%impl_witness]
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [concrete = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl: %T2.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Convert.decl: %Convert.type.c41 = fn_decl @Convert.2 [concrete = constants.%Convert.f35] {
// CHECK:STDOUT:     %self.patt: %T2 = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %T2 = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %T1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T1 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %self.param: %T2 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @impl.%T2.ref [concrete = constants.%T2]
// CHECK:STDOUT:     %self: %T2 = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .Convert = %Convert.decl
// CHECK:STDOUT:   witness = file.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.1 [concrete = constants.%F.d17] {
// CHECK:STDOUT:     %self.patt: %Base = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Base = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %T1 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T1 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %self.param: %Base = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Base [concrete = constants.%Base]
// CHECK:STDOUT:     %self: %Base = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <vtable> = vtable (%F.decl) [concrete = constants.%.5ee]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: %T2 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %T2 = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T2.ref: type = name_ref T2, file.%T2.decl [concrete = constants.%T2]
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %return.param: ref %T2 = out_param call_param1
// CHECK:STDOUT:     %return: ref %T2 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc30: <vtable> = vtable (%F.decl) [concrete = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.15c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT: fn @Convert.2[%self.param_patt: %T2]() -> %return.param_patt: %T1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %.loc12_13.2: init %T1 = class_init (), %return [concrete = constants.%T1.val]
// CHECK:STDOUT:   %.loc12_14: init %T1 = converted %.loc12_13.1, %.loc12_13.2 [concrete = constants.%T1.val]
// CHECK:STDOUT:   return %.loc12_14 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1[%self.param_patt: %Base]() -> %T1;
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %Derived]() -> %T2;
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_impl_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.type.f17: type = fn_type @F.1, @Base(%T) [symbolic]
// CHECK:STDOUT:   %F.e26: %F.type.f17 = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.f89: <vtable> = vtable (%F.e26) [symbolic]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.ea5: type = class_type @Base, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.type.d82: type = fn_type @F.1, @Base(%T1) [concrete]
// CHECK:STDOUT:   %F.d25: %F.type.d82 = struct_value () [concrete]
// CHECK:STDOUT:   %.611: <vtable> = vtable (%F.d25) [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.ea5 [concrete]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [concrete]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.ea5} [concrete]
// CHECK:STDOUT:   %complete_type.65a: <witness> = complete_type_witness %struct_type.base [concrete]
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
// CHECK:STDOUT:     %T.patt.loc7_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc7_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T1 {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T.loc7_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @Base(%T.loc7_17.2) [symbolic = %F.type (constants.%F.type.f17)]
// CHECK:STDOUT:   %F: @Base.%F.type (%F.type.f17) = struct_value () [symbolic = %F (constants.%F.e26)]
// CHECK:STDOUT:   %.loc9_1.2: <vtable> = vtable (%F) [symbolic = %.loc9_1.2 (constants.%.f89)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Base.%F.type (%F.type.f17) = fn_decl @F.1 [symbolic = @Base.%F (constants.%F.e26)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Base (%Base.370) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Base (%Base.370) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %t.patt: @F.1.%T (%T) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.1.%T (%T) = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.1.%Base (%Base.370) = value_param call_param0
// CHECK:STDOUT:       %.loc8_22.1: type = splice_block %Self.ref [symbolic = %Base (constants.%Base.370)] {
// CHECK:STDOUT:         %.loc8_22.2: type = specific_constant constants.%Base.370, @Base(constants.%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_22.2 [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Base (%Base.370) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.1.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @Base.%T.loc7_17.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.1.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9_1.1: <vtable> = vtable (%F.decl) [symbolic = %.loc9_1.2 (constants.%.f89)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:     complete_type_witness = %complete_type
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
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [concrete = constants.%F.fa3] {
// CHECK:STDOUT:     %self.patt: %Derived = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %Derived = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %t.patt: %T1 = binding_pattern t
// CHECK:STDOUT:     %t.param_patt: %T1 = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %self: %Derived = bind_name self, %self.param
// CHECK:STDOUT:     %t.param: %T1 = value_param call_param1
// CHECK:STDOUT:     %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:     %t: %T1 = bind_name t, %t.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: <vtable> = vtable (%F.decl) [concrete = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.65a]
// CHECK:STDOUT:   complete_type_witness = %complete_type
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
// CHECK:STDOUT: generic virtual fn @F.1(@Base.%T.loc7_17.1: type) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:
// CHECK:STDOUT:   virtual fn[%self.param_patt: @F.1.%Base (%Base.370)](%t.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2[%self.param_patt: %Derived](%t.param_patt: %T1);
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_17.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %Base => constants.%Base.370
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc7_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T1) {
// CHECK:STDOUT:   %T.loc7_17.2 => constants.%T1
// CHECK:STDOUT:   %T.patt.loc7_17.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.d82
// CHECK:STDOUT:   %F => constants.%F.d25
// CHECK:STDOUT:   %.loc9_1.2 => constants.%.611
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_virtual_without_self.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %F.type.ba7: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.1a5: %F.type.ba7 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %F.type.834: type = fn_type @F.2 [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type.ba7 = fn_decl @F.1 [concrete = constants.%F.1a5] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {} {}
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
// CHECK:STDOUT:   %F.decl: %F.type.834 = fn_decl @F.2 [concrete = constants.%F.a48] {} {}
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
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_addr_self_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T1: type = class_type @T1 [concrete]
// CHECK:STDOUT:   %ptr.87b: type = ptr_type %T1 [concrete]
// CHECK:STDOUT:   %F1.type.b96: type = fn_type @F1.1 [concrete]
// CHECK:STDOUT:   %F1.765: %F1.type.b96 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [concrete]
// CHECK:STDOUT:   %.278: <vtable> = vtable (%F1.765) [concrete]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [concrete]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [concrete]
// CHECK:STDOUT:   %T2: type = class_type @T2 [concrete]
// CHECK:STDOUT:   %T2.elem: type = unbound_element_type %T2, %T1 [concrete]
// CHECK:STDOUT:   %ptr.63e: type = ptr_type %T2 [concrete]
// CHECK:STDOUT:   %F1.type.b0d: type = fn_type @F1.2 [concrete]
// CHECK:STDOUT:   %F1.0ce: %F1.type.b0d = struct_value () [concrete]
// CHECK:STDOUT:   %.20a: <vtable> = vtable (%F1.0ce) [concrete]
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
// CHECK:STDOUT:   %F1.decl: %F1.type.b96 = fn_decl @F1.1 [concrete = constants.%F1.765] {
// CHECK:STDOUT:     %self.patt: %ptr.87b = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.87b = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.87b = value_param call_param0
// CHECK:STDOUT:     %.loc5: type = splice_block %ptr [concrete = constants.%ptr.87b] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%T1 [concrete = constants.%T1]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.87b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.87b = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F1.decl) [concrete = constants.%.278]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [concrete = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T1
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @T2 {
// CHECK:STDOUT:   %T1.ref: type = name_ref T1, file.%T1.decl [concrete = constants.%T1]
// CHECK:STDOUT:   %.loc9: %T2.elem = base_decl %T1.ref, element0 [concrete]
// CHECK:STDOUT:   %F1.decl: %F1.type.b0d = fn_decl @F1.2 [concrete = constants.%F1.0ce] {
// CHECK:STDOUT:     %self.patt: %ptr.63e = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %ptr.63e = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %.loc17_14: auto = addr_pattern %self.param_patt
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.63e = value_param call_param0
// CHECK:STDOUT:     %.loc17_29: type = splice_block %ptr [concrete = constants.%ptr.63e] {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%T2 [concrete = constants.%T2]
// CHECK:STDOUT:       %ptr: type = ptr_type %Self.ref [concrete = constants.%ptr.63e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self: %ptr.63e = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc18: <vtable> = vtable (%F1.decl) [concrete = constants.%.20a]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.e14]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%T2
// CHECK:STDOUT:   .T1 = <poisoned>
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F1 = %F1.decl
// CHECK:STDOUT:   extend %T1.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F1.1[%self.param_patt: %ptr.87b]();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F1.2[addr %self.param_patt: %ptr.63e]();
// CHECK:STDOUT:
