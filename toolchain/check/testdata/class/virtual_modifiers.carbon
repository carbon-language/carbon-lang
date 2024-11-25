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

  impl fn I();
}

abstract class Abstract {
  abstract fn J();

  virtual fn K();

  impl fn L();
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

// --- init_members.carbon

package InitMembers;

base class Base {
  virtual fn F();
  var m1: i32;
  var m2: i32;
}

fn F() {
  var v: Base = {.m2 = 3, .m1 = 5};
}

// CHECK:STDOUT: --- modifiers.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %I.type: type = fn_type @I [template]
// CHECK:STDOUT:   %I: %I.type = struct_value () [template]
// CHECK:STDOUT:   %.1: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.2: type = struct_type {.<vptr>: %.1} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %Abstract: type = class_type @Abstract [template]
// CHECK:STDOUT:   %J.type: type = fn_type @J [template]
// CHECK:STDOUT:   %J: %J.type = struct_value () [template]
// CHECK:STDOUT:   %K.type: type = fn_type @K [template]
// CHECK:STDOUT:   %K: %K.type = struct_value () [template]
// CHECK:STDOUT:   %L.type: type = fn_type @L [template]
// CHECK:STDOUT:   %L: %L.type = struct_value () [template]
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
// CHECK:STDOUT:   %I.decl: %I.type = fn_decl @I [template = constants.%I] {} {}
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .H = %H.decl
// CHECK:STDOUT:   .I = %I.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Abstract {
// CHECK:STDOUT:   %J.decl: %J.type = fn_decl @J [template = constants.%J] {} {}
// CHECK:STDOUT:   %K.decl: %K.type = fn_decl @K [template = constants.%K] {} {}
// CHECK:STDOUT:   %L.decl: %L.type = fn_decl @L [template = constants.%L] {} {}
// CHECK:STDOUT:   %.loc16: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Abstract
// CHECK:STDOUT:   .J = %J.decl
// CHECK:STDOUT:   .K = %K.decl
// CHECK:STDOUT:   .L = %L.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @H();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @I();
// CHECK:STDOUT:
// CHECK:STDOUT: abstract fn @J();
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @K();
// CHECK:STDOUT:
// CHECK:STDOUT: impl fn @L();
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_later_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
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
// CHECK:STDOUT:   %import_ref.2 = import_ref Modifiers//default, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Modifiers//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Modifiers//default, inst+9, unloaded
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
// CHECK:STDOUT:   %.loc8: %.5 = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.6 [template = constants.%.7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .base = %.loc8
// CHECK:STDOUT:   extend %Base.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Base {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .H = imports.%import_ref.3
// CHECK:STDOUT:   .I = imports.%import_ref.4
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
// CHECK:STDOUT:   %import_ref.2 = import_ref Modifiers//default, inst+4, unloaded
// CHECK:STDOUT:   %import_ref.3 = import_ref Modifiers//default, inst+5, unloaded
// CHECK:STDOUT:   %import_ref.4 = import_ref Modifiers//default, inst+9, unloaded
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
// CHECK:STDOUT:   .Self = imports.%import_ref.2
// CHECK:STDOUT:   .H = imports.%import_ref.3
// CHECK:STDOUT:   .I = imports.%import_ref.4
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
// CHECK:STDOUT: --- init_members.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base: type = class_type @Base [template]
// CHECK:STDOUT:   %F.type.1: type = fn_type @F.1 [template]
// CHECK:STDOUT:   %F.1: %F.type.1 = struct_value () [template]
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %.2: type = unbound_element_type %Base, %i32 [template]
// CHECK:STDOUT:   %.3: type = ptr_type <vtable> [template]
// CHECK:STDOUT:   %.4: type = struct_type {.<vptr>: %.3, .m1: %i32, .m2: %i32} [template]
// CHECK:STDOUT:   %.5: <witness> = complete_type_witness %.4 [template]
// CHECK:STDOUT:   %F.type.2: type = fn_type @F.2 [template]
// CHECK:STDOUT:   %F.2: %F.type.2 = struct_value () [template]
// CHECK:STDOUT:   %.7: Core.IntLiteral = int_value 3 [template]
// CHECK:STDOUT:   %.8: Core.IntLiteral = int_value 5 [template]
// CHECK:STDOUT:   %.9: type = struct_type {.m2: Core.IntLiteral, .m1: Core.IntLiteral} [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.33: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.34: <bound method> = bound_method %.8, %Convert.14 [template]
// CHECK:STDOUT:   %.35: <specific function> = specific_function %.34, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.36: %i32 = int_value 5 [template]
// CHECK:STDOUT:   %.37: <bound method> = bound_method %.7, %Convert.14 [template]
// CHECK:STDOUT:   %.38: <specific function> = specific_function %.37, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.39: %i32 = int_value 3 [template]
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
// CHECK:STDOUT:   %F.decl: %F.type.1 = fn_decl @F.1 [template = constants.%F.1] {} {}
// CHECK:STDOUT:   %.loc6_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc6: init type = call constants.%Int(%.loc6_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_11.2: type = value_of_initializer %int.make_type_signed.loc6 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_11.3: type = converted %int.make_type_signed.loc6, %.loc6_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc6_9: %.2 = field_decl m1, element0 [template]
// CHECK:STDOUT:   %.loc7_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:   %int.make_type_signed.loc7: init type = call constants.%Int(%.loc7_11.1) [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_11.2: type = value_of_initializer %int.make_type_signed.loc7 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_11.3: type = converted %int.make_type_signed.loc7, %.loc7_11.2 [template = constants.%i32]
// CHECK:STDOUT:   %.loc7_9: %.2 = field_decl m2, element1 [template]
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.4 [template = constants.%.5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Base
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   .m1 = %.loc6_9
// CHECK:STDOUT:   .m2 = %.loc7_9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: virtual fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Base.ref: type = name_ref Base, file.%Base.decl [template = constants.%Base]
// CHECK:STDOUT:   %v.var: ref %Base = var v
// CHECK:STDOUT:   %v: ref %Base = bind_name v, %v.var
// CHECK:STDOUT:   %.loc11_24: Core.IntLiteral = int_value 3 [template = constants.%.7]
// CHECK:STDOUT:   %.loc11_33: Core.IntLiteral = int_value 5 [template = constants.%.8]
// CHECK:STDOUT:   %.loc11_34.1: %.9 = struct_literal (%.loc11_24, %.loc11_33)
// CHECK:STDOUT:   %.loc11_34.2: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_34.3: <bound method> = bound_method %.loc11_33, %.loc11_34.2 [template = constants.%.34]
// CHECK:STDOUT:   %.loc11_34.4: <specific function> = specific_function %.loc11_34.3, @Convert.2(constants.%.1) [template = constants.%.35]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.1: init %i32 = call %.loc11_34.4(%.loc11_33) [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_34.5: init %i32 = converted %.loc11_33, %int.convert_checked.loc11_34.1 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_34.6: ref %i32 = class_element_access %v.var, element2
// CHECK:STDOUT:   %.loc11_34.7: init %i32 = initialize_from %.loc11_34.5 to %.loc11_34.6 [template = constants.%.36]
// CHECK:STDOUT:   %.loc11_34.8: %Convert.type.2 = interface_witness_access constants.%.33, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc11_34.9: <bound method> = bound_method %.loc11_24, %.loc11_34.8 [template = constants.%.37]
// CHECK:STDOUT:   %.loc11_34.10: <specific function> = specific_function %.loc11_34.9, @Convert.2(constants.%.1) [template = constants.%.38]
// CHECK:STDOUT:   %int.convert_checked.loc11_34.2: init %i32 = call %.loc11_34.10(%.loc11_24) [template = constants.%.39]
// CHECK:STDOUT:   %.loc11_34.11: init %i32 = converted %.loc11_24, %int.convert_checked.loc11_34.2 [template = constants.%.39]
// CHECK:STDOUT:   %.loc11_34.12: ref %i32 = class_element_access %v.var, element1
// CHECK:STDOUT:   %.loc11_34.13: init %i32 = initialize_from %.loc11_34.11 to %.loc11_34.12 [template = constants.%.39]
// CHECK:STDOUT:   %.loc11_34.14: init %Base = class_init (<error>, %.loc11_34.7, %.loc11_34.13), %v.var [template = <error>]
// CHECK:STDOUT:   %.loc11_35: init %Base = converted %.loc11_34.1, %.loc11_34.14 [template = <error>]
// CHECK:STDOUT:   assign %v.var, %.loc11_35
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
