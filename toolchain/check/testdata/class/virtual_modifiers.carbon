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
  virtual fn H();
}

abstract class Abstract {
  abstract fn J();

  virtual fn K();
}

// --- override_import.carbon

package OverrideImport;

import Modifiers;

class Derived {
  extend base: Modifiers.Base;
  impl fn H();
}

// --- todo_fail_later_base.carbon

package FailLaterBase;

import Modifiers;

base class Derived {
  virtual fn F();
  extend base: Modifiers.Base;
}

// --- init.carbon

package Init;

import Modifiers;

fn F() {
  var v: Modifiers.Base = {};
}

// --- impl_abstract.carbon

package ImplAbstract;

abstract class A1 {
  virtual fn F();
}

abstract class A2 {
  extend base: A1;
  impl fn F();
}

// --- impl_base.carbon

package ImplBase;

base class B1 {
  virtual fn F();
}

base class B2 {
  extend base: B1;
  impl fn F();
}

class C {
  extend base: B2;
  impl fn F();
}

// --- fail_modifiers.carbon

package FailModifiers;

class C {
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+4]]:3: error: impl without base class [ImplWithoutBase]
  // CHECK:STDERR:   impl fn F();
  // CHECK:STDERR:   ^~~~~~~~~~~~
  // CHECK:STDERR:
  impl fn F();
}

// --- init_members.carbon

package InitMembers;

base class Base {
  var m1: i32;
  var m2: i32;

  virtual fn F();
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

package ImplWithoutBaseDeclaration;

base class Base {
}

class Derived {
  extend base: Base;
  impl fn F();
}

// --- abstract_impl.carbon

package AbstractImpl;

abstract class AbstractBase {
  abstract fn F();
}

abstract class AbstractIntermediate {
  extend base: AbstractBase;
}

class Derived {
  extend base: AbstractIntermediate;
  impl fn F();
}

// --- virtual_impl.carbon

package VirtualImpl;

base class VirtualBase {
  virtual fn F();
}

base class VirtualIntermediate {
  extend base: VirtualBase;
}

class Derived {
  extend base: VirtualIntermediate;
  impl fn F();
}

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.c3d: <vtable> = vtable (%H) [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %.2b2: <vtable> = vtable (%J, %K) [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Abstract = %Abstract.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Abstract.decl: type = class_decl @Abstract [template = constants.%Abstract] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%H.decl) [template = constants.%.c3d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {} {}
// CHECK:STDOUT:   %.loc12: <vtable> = vtable (%J.decl, %K.decl) [template = constants.%.2b2]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @J();
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @K();
// CHECK:STDOUT:
// CHECK:STDOUT: --- override_import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %H.type.dba: type = fn_type @H.1 [template]
// CHECK:STDOUT:   %H.bce: %H.type.dba = struct_value () [template]
// CHECK:STDOUT:   %.dce: <vtable> = vtable (%H.bce) [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [template] {
// CHECK:STDOUT:     .Base = %import_ref.917
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.917: type = import_ref Modifiers//default, Base, loaded [template = constants.%Base]
// CHECK:STDOUT:   %import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [template = constants.%complete_type.513]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [template = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%import_ref.917 [template = constants.%Base]
// CHECK:STDOUT:   %.loc7: %Derived.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %H.decl: %H.type.dba = fn_decl @H.1 [template = constants.%H.bce] {} {}
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (%H.decl) [template = constants.%.dce]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @H.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.2() [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.02f: <vtable> = vtable (%H, %F) [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %complete_type.0e2: <witness> = complete_type_witness %struct_type.base [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [template] {
// CHECK:STDOUT:     .Base = %import_ref.917
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.917: type = import_ref Modifiers//default, Base, loaded [template = constants.%Base]
// CHECK:STDOUT:   %import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [template = constants.%complete_type.513]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [template = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%import_ref.917 [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (constants.%H, %F.decl) [template = constants.%.02f]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.0e2]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() [from "modifiers.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [template] {
// CHECK:STDOUT:     .Base = %import_ref.917
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.917: type = import_ref Modifiers//default, Base, loaded [template = constants.%Base]
// CHECK:STDOUT:   %import_ref.05e: <witness> = import_ref Modifiers//default, loc6_1, loaded [template = constants.%complete_type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Modifiers = imports.%Modifiers
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Modifiers.import = import Modifiers
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base [from "modifiers.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.05e
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
// CHECK:STDOUT:   %.loc7_28.3: ref %ptr.454 = vtable_ptr
// CHECK:STDOUT:   %.loc7_28.4: init %ptr.454 = initialize_from %.loc7_28.3 to %.loc7_28.2
// CHECK:STDOUT:   %.loc7_28.5: init %Base = class_init (%.loc7_28.4), %v.var
// CHECK:STDOUT:   %.loc7_3.2: init %Base = converted %.loc7_28.1, %.loc7_28.5
// CHECK:STDOUT:   assign %v.var, %.loc7_3.2
// CHECK:STDOUT:   %.loc7_19: type = splice_block %Base.ref [template = constants.%Base] {
// CHECK:STDOUT:     %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [template = imports.%Modifiers]
// CHECK:STDOUT:     %Base.ref: type = name_ref Base, imports.%import_ref.917 [template = constants.%Base]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %F.type.13a: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.df5: %F.type.13a = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.593: <vtable> = vtable (%F.df5) [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %A2.elem: type = unbound_element_type %A2, %A1 [template]
// CHECK:STDOUT:   %F.type.4ae: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.1d5: %F.type.4ae = struct_value () [template]
// CHECK:STDOUT:   %.943: <vtable> = vtable (%F.1d5) [template]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %A1} [template]
// CHECK:STDOUT:   %complete_type.a6f: <witness> = complete_type_witness %struct_type.base [template]
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
// CHECK:STDOUT:     .A1 = %A1.decl
// CHECK:STDOUT:     .A2 = %A2.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A1.decl: type = class_decl @A1 [template = constants.%A1] {} {}
// CHECK:STDOUT:   %A2.decl: type = class_decl @A2 [template = constants.%A2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A1 {
// CHECK:STDOUT:   %F.decl: %F.type.13a = fn_decl @F.1 [template = constants.%F.df5] {} {}
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [template = constants.%.593]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %A1.ref: type = name_ref A1, file.%A1.decl [template = constants.%A1]
// CHECK:STDOUT:   %.loc9: %A2.elem = base_decl %A1.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.4ae = fn_decl @F.2 [template = constants.%F.1d5] {} {}
// CHECK:STDOUT:   %.loc11: <vtable> = vtable (%F.decl) [template = constants.%.943]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [template = constants.%complete_type.a6f]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B1: type = class_type @B1 [template]
// CHECK:STDOUT:   %F.type.e4c: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.8f5: %F.type.e4c = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.bc5: <vtable> = vtable (%F.8f5) [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %B2.elem: type = unbound_element_type %B2, %B1 [template]
// CHECK:STDOUT:   %F.type.b26: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.d48: %F.type.b26 = struct_value () [template]
// CHECK:STDOUT:   %.579: <vtable> = vtable (%F.d48) [template]
// CHECK:STDOUT:   %struct_type.base.508: type = struct_type {.base: %B1} [template]
// CHECK:STDOUT:   %complete_type.5ac: <witness> = complete_type_witness %struct_type.base.508 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %B2 [template]
// CHECK:STDOUT:   %F.type.c29: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.437: %F.type.c29 = struct_value () [template]
// CHECK:STDOUT:   %.5f6: <vtable> = vtable (%F.437) [template]
// CHECK:STDOUT:   %struct_type.base.421: type = struct_type {.base: %B2} [template]
// CHECK:STDOUT:   %complete_type.066: <witness> = complete_type_witness %struct_type.base.421 [template]
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
// CHECK:STDOUT:     .B1 = %B1.decl
// CHECK:STDOUT:     .B2 = %B2.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %B1.decl: type = class_decl @B1 [template = constants.%B1] {} {}
// CHECK:STDOUT:   %B2.decl: type = class_decl @B2 [template = constants.%B2] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B1 {
// CHECK:STDOUT:   %F.decl: %F.type.e4c = fn_decl @F.1 [template = constants.%F.8f5] {} {}
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [template = constants.%.bc5]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [template = constants.%B1]
// CHECK:STDOUT:   %.loc9: %B2.elem = base_decl %B1.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.b26 = fn_decl @F.2 [template = constants.%F.d48] {} {}
// CHECK:STDOUT:   %.loc11: <vtable> = vtable (%F.decl) [template = constants.%.579]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.508 [template = constants.%complete_type.5ac]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [template = constants.%B2]
// CHECK:STDOUT:   %.loc14: %C.elem = base_decl %B2.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.c29 = fn_decl @F.3 [template = constants.%F.437] {} {}
// CHECK:STDOUT:   %.loc16: <vtable> = vtable (%F.decl) [template = constants.%.5f6]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.421 [template = constants.%complete_type.066]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.3();
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.f2b: <vtable> = vtable () [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc10: <vtable> = vtable () [template = constants.%.f2b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %F.type.7c6: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.d17: %F.type.7c6 = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.5ee: <vtable> = vtable (%F.d17) [template]
// CHECK:STDOUT:   %struct_type.vptr.m1.m2: type = struct_type {.<vptr>: %ptr.454, .m1: %i32, .m2: %i32} [template]
// CHECK:STDOUT:   %complete_type.cf7: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [template]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [template]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%import_ref.a5b), @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.1(%int_32) [template]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [template]
// CHECK:STDOUT:   %Convert.bound.b30: <bound method> = bound_method %int_3.1ba, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.b42: <specific function> = specific_function %Convert.bound.b30, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_3.822: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %struct_type.m2.m1.68c: type = struct_type {.m2: %i32, .m1: %i32} [template]
// CHECK:STDOUT:   %int_5.64b: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %struct_type.m2.m1.5f2: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.bound.4e6: <bound method> = bound_method %int_5.64b, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.ba9: <specific function> = specific_function %Convert.bound.4e6, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_5.0f6: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %Convert.bound.ac3: <bound method> = bound_method %int_4.0c1, %Convert.956 [template]
// CHECK:STDOUT:   %Convert.specific_fn.450: <specific function> = specific_function %Convert.bound.ac3, @Convert.2(%int_32) [template]
// CHECK:STDOUT:   %int_4.940: %i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.485
// CHECK:STDOUT:     .ImplicitAs = %import_ref.d44
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [template = constants.%F.c41] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5_9: %Base.elem = field_decl m1, element1 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc5_3: %Base.elem = var_pattern %.loc5_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc5: ref %Base.elem = var <invalid>
// CHECK:STDOUT:   %.loc6_9: %Base.elem = field_decl m2, element2 [template]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc6_3: %Base.elem = var_pattern %.loc6_9
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var.loc6: ref %Base.elem = var <invalid>
// CHECK:STDOUT:   %F.decl: %F.type.7c6 = fn_decl @F.1 [template = constants.%F.d17] {} {}
// CHECK:STDOUT:   %.loc9: <vtable> = vtable (%F.decl) [template = constants.%.5ee]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.m1.m2 [template = constants.%complete_type.cf7]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %i32 = binding_pattern i
// CHECK:STDOUT:     %.loc12_3.1: %i32 = var_pattern %i.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %i.var: ref %i32 = var i
// CHECK:STDOUT:   %int_3.loc12: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %impl.elem0.loc12: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc12: <bound method> = bound_method %int_3.loc12, %impl.elem0.loc12 [template = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %Convert.specific_fn.loc12: <specific function> = specific_function %Convert.bound.loc12, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b42]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %Convert.specific_fn.loc12(%int_3.loc12) [template = constants.%int_3.822]
// CHECK:STDOUT:   %.loc12_3.2: init %i32 = converted %int_3.loc12, %int.convert_checked.loc12 [template = constants.%int_3.822]
// CHECK:STDOUT:   assign %i.var, %.loc12_3.2
// CHECK:STDOUT:   %.loc12_10: type = splice_block %i32 [template = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
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
// CHECK:STDOUT:   %.loc14_35.3: ref %ptr.454 = vtable_ptr
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
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b1: ref %Base = bind_name b1, %b1.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b2.patt: %Base = binding_pattern b2
// CHECK:STDOUT:     %.loc15_3.1: %Base = var_pattern %b2.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b2.var: ref %Base = var b2
// CHECK:STDOUT:   %int_3.loc15: Core.IntLiteral = int_value 3 [template = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_5: Core.IntLiteral = int_value 5 [template = constants.%int_5.64b]
// CHECK:STDOUT:   %.loc15_35.1: %struct_type.m2.m1.5f2 = struct_literal (%int_3.loc15, %int_5)
// CHECK:STDOUT:   %.loc15_35.2: ref %ptr.454 = class_element_access %b2.var, element0
// CHECK:STDOUT:   %.loc15_35.3: ref %ptr.454 = vtable_ptr
// CHECK:STDOUT:   %.loc15_35.4: init %ptr.454 = initialize_from %.loc15_35.3 to %.loc15_35.2
// CHECK:STDOUT:   %impl.elem0.loc15_35.1: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc15_35.1: <bound method> = bound_method %int_5, %impl.elem0.loc15_35.1 [template = constants.%Convert.bound.4e6]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_35.1: <specific function> = specific_function %Convert.bound.loc15_35.1, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.ba9]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.1: init %i32 = call %Convert.specific_fn.loc15_35.1(%int_5) [template = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.5: init %i32 = converted %int_5, %int.convert_checked.loc15_35.1 [template = constants.%int_5.0f6]
// CHECK:STDOUT:   %.loc15_35.6: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc15_35.7: init %i32 = initialize_from %.loc15_35.5 to %.loc15_35.6 [template = constants.%int_5.0f6]
// CHECK:STDOUT:   %impl.elem0.loc15_35.2: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc15_35.2: <bound method> = bound_method %int_3.loc15, %impl.elem0.loc15_35.2 [template = constants.%Convert.bound.b30]
// CHECK:STDOUT:   %Convert.specific_fn.loc15_35.2: <specific function> = specific_function %Convert.bound.loc15_35.2, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.b42]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.2: init %i32 = call %Convert.specific_fn.loc15_35.2(%int_3.loc15) [template = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.8: init %i32 = converted %int_3.loc15, %int.convert_checked.loc15_35.2 [template = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.9: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc15_35.10: init %i32 = initialize_from %.loc15_35.8 to %.loc15_35.9 [template = constants.%int_3.822]
// CHECK:STDOUT:   %.loc15_35.11: init %Base = class_init (%.loc15_35.4, %.loc15_35.7, %.loc15_35.10), %b2.var
// CHECK:STDOUT:   %.loc15_3.2: init %Base = converted %.loc15_35.1, %.loc15_35.11
// CHECK:STDOUT:   assign %b2.var, %.loc15_3.2
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b2: ref %Base = bind_name b2, %b2.var
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %Base.elem = name_ref m2, @Base.%.loc6_9 [template = @Base.%.loc6_9]
// CHECK:STDOUT:   %.loc18_5: ref %i32 = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %int_4: Core.IntLiteral = int_value 4 [template = constants.%int_4.0c1]
// CHECK:STDOUT:   %impl.elem0.loc18: %Convert.type.1b6 = impl_witness_access constants.%impl_witness.d39, element0 [template = constants.%Convert.956]
// CHECK:STDOUT:   %Convert.bound.loc18: <bound method> = bound_method %int_4, %impl.elem0.loc18 [template = constants.%Convert.bound.ac3]
// CHECK:STDOUT:   %Convert.specific_fn.loc18: <specific function> = specific_function %Convert.bound.loc18, @Convert.2(constants.%int_32) [template = constants.%Convert.specific_fn.450]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %Convert.specific_fn.loc18(%int_4) [template = constants.%int_4.940]
// CHECK:STDOUT:   %.loc18_9: init %i32 = converted %int_4, %int.convert_checked.loc18 [template = constants.%int_4.940]
// CHECK:STDOUT:   assign %.loc18_5, %.loc18_9
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_impl_without_base_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.f2b: <vtable> = vtable () [template]
// CHECK:STDOUT:   %struct_type.vptr.base: type = struct_type {.<vptr>: %ptr.454, .base: %Base} [template]
// CHECK:STDOUT:   %complete_type.336: <witness> = complete_type_witness %struct_type.vptr.base [template]
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
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: type = class_decl @Base [template = constants.%Base] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %Derived.elem = base_decl %Base.ref, element1 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc10: <vtable> = vtable () [template = constants.%.f2b]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr.base [template = constants.%complete_type.336]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- abstract_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %AbstractBase: type = class_type @AbstractBase [template]
// CHECK:STDOUT:   %F.type.85b: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.6e9: %F.type.85b = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.6ec: <vtable> = vtable (%F.6e9) [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %AbstractIntermediate: type = class_type @AbstractIntermediate [template]
// CHECK:STDOUT:   %AbstractIntermediate.elem: type = unbound_element_type %AbstractIntermediate, %AbstractBase [template]
// CHECK:STDOUT:   %struct_type.base.efd: type = struct_type {.base: %AbstractBase} [template]
// CHECK:STDOUT:   %complete_type.2d3: <witness> = complete_type_witness %struct_type.base.efd [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %AbstractIntermediate [template]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [template]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [template]
// CHECK:STDOUT:   %struct_type.base.da5: type = struct_type {.base: %AbstractIntermediate} [template]
// CHECK:STDOUT:   %complete_type.f8c: <witness> = complete_type_witness %struct_type.base.da5 [template]
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
// CHECK:STDOUT:     .AbstractBase = %AbstractBase.decl
// CHECK:STDOUT:     .AbstractIntermediate = %AbstractIntermediate.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %AbstractBase.decl: type = class_decl @AbstractBase [template = constants.%AbstractBase] {} {}
// CHECK:STDOUT:   %AbstractIntermediate.decl: type = class_decl @AbstractIntermediate [template = constants.%AbstractIntermediate] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractBase {
// CHECK:STDOUT:   %F.decl: %F.type.85b = fn_decl @F.1 [template = constants.%F.6e9] {} {}
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [template = constants.%.6ec]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @AbstractIntermediate {
// CHECK:STDOUT:   %AbstractBase.ref: type = name_ref AbstractBase, file.%AbstractBase.decl [template = constants.%AbstractBase]
// CHECK:STDOUT:   %.loc9: %AbstractIntermediate.elem = base_decl %AbstractBase.ref, element0 [template]
// CHECK:STDOUT:   %.loc10: <vtable> = vtable (constants.%F.6e9) [template = constants.%.6ec]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.efd [template = constants.%complete_type.2d3]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %AbstractIntermediate.ref: type = name_ref AbstractIntermediate, file.%AbstractIntermediate.decl [template = constants.%AbstractIntermediate]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %AbstractIntermediate.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [template = constants.%F.fa3] {} {}
// CHECK:STDOUT:   %.loc15: <vtable> = vtable (%F.decl) [template = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.da5 [template = constants.%complete_type.f8c]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2();
// CHECK:STDOUT:
// CHECK:STDOUT: --- virtual_impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %VirtualBase: type = class_type @VirtualBase [template]
// CHECK:STDOUT:   %F.type.e62: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.3e7: %F.type.e62 = struct_value () [template]
// CHECK:STDOUT:   %ptr.454: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.def: <vtable> = vtable (%F.3e7) [template]
// CHECK:STDOUT:   %struct_type.vptr: type = struct_type {.<vptr>: %ptr.454} [template]
// CHECK:STDOUT:   %complete_type.513: <witness> = complete_type_witness %struct_type.vptr [template]
// CHECK:STDOUT:   %VirtualIntermediate: type = class_type @VirtualIntermediate [template]
// CHECK:STDOUT:   %VirtualIntermediate.elem: type = unbound_element_type %VirtualIntermediate, %VirtualBase [template]
// CHECK:STDOUT:   %struct_type.base.61e: type = struct_type {.base: %VirtualBase} [template]
// CHECK:STDOUT:   %complete_type.f09: <witness> = complete_type_witness %struct_type.base.61e [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %VirtualIntermediate [template]
// CHECK:STDOUT:   %F.type.5da: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.fa3: %F.type.5da = struct_value () [template]
// CHECK:STDOUT:   %.88d: <vtable> = vtable (%F.fa3) [template]
// CHECK:STDOUT:   %struct_type.base.43c: type = struct_type {.base: %VirtualIntermediate} [template]
// CHECK:STDOUT:   %complete_type.fa6: <witness> = complete_type_witness %struct_type.base.43c [template]
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
// CHECK:STDOUT:     .VirtualBase = %VirtualBase.decl
// CHECK:STDOUT:     .VirtualIntermediate = %VirtualIntermediate.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %VirtualBase.decl: type = class_decl @VirtualBase [template = constants.%VirtualBase] {} {}
// CHECK:STDOUT:   %VirtualIntermediate.decl: type = class_decl @VirtualIntermediate [template = constants.%VirtualIntermediate] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualBase {
// CHECK:STDOUT:   %F.decl: %F.type.e62 = fn_decl @F.1 [template = constants.%F.3e7] {} {}
// CHECK:STDOUT:   %.loc6: <vtable> = vtable (%F.decl) [template = constants.%.def]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.vptr [template = constants.%complete_type.513]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @VirtualIntermediate {
// CHECK:STDOUT:   %VirtualBase.ref: type = name_ref VirtualBase, file.%VirtualBase.decl [template = constants.%VirtualBase]
// CHECK:STDOUT:   %.loc9: %VirtualIntermediate.elem = base_decl %VirtualBase.ref, element0 [template]
// CHECK:STDOUT:   %.loc10: <vtable> = vtable (constants.%F.3e7) [template = constants.%.def]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.61e [template = constants.%complete_type.f09]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %VirtualIntermediate.ref: type = name_ref VirtualIntermediate, file.%VirtualIntermediate.decl [template = constants.%VirtualIntermediate]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %VirtualIntermediate.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.5da = fn_decl @F.2 [template = constants.%F.fa3] {} {}
// CHECK:STDOUT:   %.loc15: <vtable> = vtable (%F.decl) [template = constants.%.88d]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.43c [template = constants.%complete_type.fa6]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F.2();
// CHECK:STDOUT:
