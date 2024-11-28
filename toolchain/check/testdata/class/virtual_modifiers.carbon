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
  // CHECK:STDERR: fail_modifiers.carbon:[[@LINE+3]]:3: error: impl without base class [ImplWithoutBase]
  // CHECK:STDERR:   impl fn F();
  // CHECK:STDERR:   ^~~~~~~~~~~~
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

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
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
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   complete_type_witness = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {} {}
// CHECK:STDOUT:   %.loc12: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   complete_type_witness = %.loc12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @J();
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @K();
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %Base} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [template] {
// CHECK:STDOUT:     .Base = %import_ref.1
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Modifiers//default, inst+3, loaded [template = constants.%Base]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Modifiers//default, inst+11, loaded [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3 = import_ref Modifiers//default, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Modifiers//default, inst+5, unloaded
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
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%import_ref.1 [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %.5 = base_decl %Base.ref, element0 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .H = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- init.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %.5: type = struct_type {} [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Modifiers: <namespace> = namespace file.%Modifiers.import, [template] {
// CHECK:STDOUT:     .Base = %import_ref.1
// CHECK:STDOUT:     import Modifiers//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.1: type = import_ref Modifiers//default, inst+3, loaded [template = constants.%Base]
// CHECK:STDOUT:   %import_ref.2: <witness> = import_ref Modifiers//default, inst+11, loaded [template = constants.%.3]
// CHECK:STDOUT:   %import_ref.3 = import_ref Modifiers//default, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Modifiers//default, inst+5, unloaded
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
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.3
// CHECK:STDOUT:   .H = imports.%import_ref.4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Modifiers.ref: <namespace> = name_ref Modifiers, imports.%Modifiers [template = imports.%Modifiers]
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, imports.%import_ref.1 [template = constants.%Base]
// CHECK:STDOUT:   %v.var: ref %Base = var v
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   %.loc7_28.1: %.5 = struct_literal ()
// CHECK:STDOUT:   %.loc7_28.2: init %Base = class_init (<error>), %v.var [template = <error>]
// CHECK:STDOUT:   %.loc7_29: init %Base = converted %.loc7_28.1, %.loc7_28.2 [template = <error>]
// CHECK:STDOUT:   assign %v.var, %.loc7_29
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- impl_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A1: type = class_type @A1 [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %A2: type = class_type @A2 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %A2, %A1 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %A1} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
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
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @A2 {
// CHECK:STDOUT:   %A1.ref: type = name_ref A1, file.%A1.decl [template = constants.%A1]
// CHECK:STDOUT:   %.loc9: %.5 = base_decl %A1.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%A2
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %A1.ref
// CHECK:STDOUT:   complete_type_witness = %.loc11
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
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %B2: type = class_type @B2 [template]
// CHECK:STDOUT:   %.5: type = unbound_element_type %B2, %B1 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: type = struct_type {.base: %B1} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %.10: type = unbound_element_type %C, %B2 [template]
// CHECK:STDOUT:   %F.type.3: type = fn_type @F.3 [template]
// CHECK:STDOUT:   %F.3: %F.type.3 = struct_value () [template]
// CHECK:STDOUT:   %.11: type = struct_type {.base: %B2} [template]
// CHECK:STDOUT:   %.12: <witness> = complete_type_witness %.11 [template]
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
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B1
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %.loc6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B2 {
// CHECK:STDOUT:   %B1.ref: type = name_ref B1, file.%B1.decl [template = constants.%B1]
// CHECK:STDOUT:   %.loc9: %.5 = base_decl %B1.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT:   %.loc11: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B2
// CHECK:STDOUT:   .base = %.loc9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B1.ref
// CHECK:STDOUT:   complete_type_witness = %.loc11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %B2.ref: type = name_ref B2, file.%B2.decl [template = constants.%B2]
// CHECK:STDOUT:   %.loc14: %.10 = base_decl %B2.ref, element0 [template]
// CHECK:STDOUT:   %F.decl: %F.type.3 = fn_decl @F.3 [template = constants.%F.3] {} {}
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.11 [template = constants.%.12]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %B2.ref
// CHECK:STDOUT:   complete_type_witness = %.loc16
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
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
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
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.3: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.4: type = struct_type {.<vptr>: %.3, .m1: %i32, .m2: %i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.6: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.30: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.31: <bound method> = bound_method %.6, %Convert.14 [template]
// CHECK:STDOUT:   %.32: <specific function> = specific_function %.31, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.33: %i32 = int_value 3 [template]
// CHECK:STDOUT:   %.35: type = struct_type {.m2: %i32, .m1: %i32} [template]
// CHECK:STDOUT:   %.36: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.37: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [template]
// CHECK:STDOUT:   %.38: <bound method> = bound_method %.36, %Convert.14 [template]
// CHECK:STDOUT:   %.39: <specific function> = specific_function %.38, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.40: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.41: Core.IntLiteral = int_value 4 [template]
// CHECK:STDOUT:   %.42: <bound method> = bound_method %.41, %Convert.14 [template]
// CHECK:STDOUT:   %.43: <specific function> = specific_function %.42, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.44: %i32 = int_value 4 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
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
// CHECK:STDOUT:   %F.decl: %F.type.2 = fn_decl @F.2 [template = constants.%F.2] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT:   %.loc5_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc5: init type = call constants.%Int(%.loc5_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_11.2: type = value_of_initializer %int.make_type_signed.loc5 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_11.3: type = converted %int.make_type_signed.loc5, %.loc5_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc5_9: %.2 = field_decl m1, element1 [template]
// CHECK:STDOUT:   %.loc6_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_11.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_11.3: type = converted %int.make_type_signed.loc6, %.loc6_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_9: %.2 = field_decl m2, element2 [template]
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .m1 = %.loc5_9
// CHECK:STDOUT:   .m2 = %.loc6_9
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   complete_type_witness = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc12_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed: init type = call constants.%Int(%.loc12_10.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:   %.loc12_10.3: type = converted %int.make_type_signed, %.loc12_10.2 [template = constants.%i32]
// CHECK:STDOUT:   %i.var: ref %i32 = var i
// CHECK:STDOUT:   %i: ref %i32 = bind_name i, %i.var
// CHECK:STDOUT:   %.loc12_16: Core.IntLiteral = int_value 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc12_17.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc12_17.2: <bound method> = bound_method %.loc12_16, %.loc12_17.1 [template = constants.%.31]
// CHECK:STDOUT:   %.loc12_17.3: <specific function> = specific_function %.loc12_17.2, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc12: init %i32 = call %.loc12_17.3(%.loc12_16) [template = constants.%.33]
// CHECK:STDOUT:   %.loc12_17.4: init %i32 = converted %.loc12_16, %int.convert_checked.loc12 [template = constants.%.33]
// CHECK:STDOUT:   assign %i.var, %.loc12_17.4
// CHECK:STDOUT:   %Base.ref.loc14: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b1.var: ref %Base = var b1
// CHECK:STDOUT:   %b1: ref %Base = bind_name b1, %b1.var
// CHECK:STDOUT:   %i.ref.loc14_25: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %i.ref.loc14_34: ref %i32 = name_ref i, %i
// CHECK:STDOUT:   %.loc14_35.1: %.35 = struct_literal (%i.ref.loc14_25, %i.ref.loc14_34)
// CHECK:STDOUT:   %.loc14_34: %i32 = bind_value %i.ref.loc14_34
// CHECK:STDOUT:   %.loc14_35.2: ref %i32 = class_element_access %b1.var, element2
// CHECK:STDOUT:   %.loc14_35.3: init %i32 = initialize_from %.loc14_34 to %.loc14_35.2
// CHECK:STDOUT:   %.loc14_25: %i32 = bind_value %i.ref.loc14_25
// CHECK:STDOUT:   %.loc14_35.4: ref %i32 = class_element_access %b1.var, element1
// CHECK:STDOUT:   %.loc14_35.5: init %i32 = initialize_from %.loc14_25 to %.loc14_35.4
// CHECK:STDOUT:   %.loc14_35.6: init %Base = class_init (<error>, %.loc14_35.3, %.loc14_35.5), %b1.var
// CHECK:STDOUT:   %.loc14_36: init %Base = converted %.loc14_35.1, %.loc14_35.6
// CHECK:STDOUT:   assign %b1.var, %.loc14_36
// CHECK:STDOUT:   %Base.ref.loc15: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %b2.var: ref %Base = var b2
// CHECK:STDOUT:   %b2: ref %Base = bind_name b2, %b2.var
// CHECK:STDOUT:   %.loc15_25: Core.IntLiteral = int_value 3 [template = constants.%.6]
// CHECK:STDOUT:   %.loc15_34: Core.IntLiteral = int_value 5 [template = constants.%.36]
// CHECK:STDOUT:   %.loc15_35.1: %.37 = struct_literal (%.loc15_25, %.loc15_34)
// CHECK:STDOUT:   %.loc15_35.2: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_35.3: <bound method> = bound_method %.loc15_34, %.loc15_35.2 [template = constants.%.38]
// CHECK:STDOUT:   %.loc15_35.4: <specific function> = specific_function %.loc15_35.3, @Convert.2(constants.%.1) [template = constants.%.39]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.1: init %i32 = call %.loc15_35.4(%.loc15_34) [template = constants.%.40]
// CHECK:STDOUT:   %.loc15_35.5: init %i32 = converted %.loc15_34, %int.convert_checked.loc15_35.1 [template = constants.%.40]
// CHECK:STDOUT:   %.loc15_35.6: ref %i32 = class_element_access %b2.var, element2
// CHECK:STDOUT:   %.loc15_35.7: init %i32 = initialize_from %.loc15_35.5 to %.loc15_35.6 [template = constants.%.40]
// CHECK:STDOUT:   %.loc15_35.8: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc15_35.9: <bound method> = bound_method %.loc15_25, %.loc15_35.8 [template = constants.%.31]
// CHECK:STDOUT:   %.loc15_35.10: <specific function> = specific_function %.loc15_35.9, @Convert.2(constants.%.1) [template = constants.%.32]
// CHECK:STDOUT:   %int.convert_checked.loc15_35.2: init %i32 = call %.loc15_35.10(%.loc15_25) [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_35.11: init %i32 = converted %.loc15_25, %int.convert_checked.loc15_35.2 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_35.12: ref %i32 = class_element_access %b2.var, element1
// CHECK:STDOUT:   %.loc15_35.13: init %i32 = initialize_from %.loc15_35.11 to %.loc15_35.12 [template = constants.%.33]
// CHECK:STDOUT:   %.loc15_35.14: init %Base = class_init (<error>, %.loc15_35.7, %.loc15_35.13), %b2.var [template = <error>]
// CHECK:STDOUT:   %.loc15_36: init %Base = converted %.loc15_35.1, %.loc15_35.14 [template = <error>]
// CHECK:STDOUT:   assign %b2.var, %.loc15_36
// CHECK:STDOUT:   %b1.ref: ref %Base = name_ref b1, %b1
// CHECK:STDOUT:   %m2.ref: %.2 = name_ref m2, @Base.%.loc6_9 [template = @Base.%.loc6_9]
// CHECK:STDOUT:   %.loc18_5: ref %i32 = class_element_access %b1.ref, element2
// CHECK:STDOUT:   %.loc18_11: Core.IntLiteral = int_value 4 [template = constants.%.41]
// CHECK:STDOUT:   %.loc18_9.1: %Convert.type.2 = interface_witness_access constants.%.30, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc18_9.2: <bound method> = bound_method %.loc18_11, %.loc18_9.1 [template = constants.%.42]
// CHECK:STDOUT:   %.loc18_9.3: <specific function> = specific_function %.loc18_9.2, @Convert.2(constants.%.1) [template = constants.%.43]
// CHECK:STDOUT:   %int.convert_checked.loc18: init %i32 = call %.loc18_9.3(%.loc18_11) [template = constants.%.44]
// CHECK:STDOUT:   %.loc18_9.4: init %i32 = converted %.loc18_11, %int.convert_checked.loc18 [template = constants.%.44]
// CHECK:STDOUT:   assign %.loc18_5, %.loc18_9.4
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_impl_without_base_declaration.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %.4: type = unbound_element_type %Derived, %Base [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.6: type = struct_type {.<vptr>: %.5, .base: %Base} [template]
// CHECK:STDOUT:   %.7: <witness> = complete_type_witness %.6 [template]
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
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   complete_type_witness = %.loc5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %.loc8: %.4 = base_decl %Base.ref, element1 [template]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT:   %.loc10: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT:   complete_type_witness = %.loc10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @F();
// CHECK:STDOUT:
