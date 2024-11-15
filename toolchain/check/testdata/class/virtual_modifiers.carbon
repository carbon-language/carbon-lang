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
