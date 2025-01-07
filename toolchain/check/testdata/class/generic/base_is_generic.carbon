// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/base_is_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/base_is_generic.carbon

// --- extend_generic_base.carbon

library "[[@TEST_NAME]]";

base class Base(T:! type) {
  var x: T;
}

class Param {
  var y: i32;
}

class Derived {
  extend base: Base(Param);
}

fn DoubleFieldAccess(d: Derived) -> i32 {
  return d.x.y;
}

// --- import.carbon

library "[[@TEST_NAME]]";

import library "extend_generic_base";

fn ImportedDoubleFieldAccess(d: Derived) -> i32 {
  return d.x.y;
}

// --- fail_todo_extend_symbolic_base.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  // CHECK:STDERR: fail_todo_extend_symbolic_base.carbon:[[@LINE+3]]:16: error: deriving from final type `T`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: T;
  // CHECK:STDERR:                ^
  extend base: T;
}

base class X {
  fn G() {}
}

fn F() {
  C(X).G();
}

// --- extend_generic_symbolic_base.carbon

library "[[@TEST_NAME]]";

base class X(U:! type) {
  fn G() -> U { return G(); }
}

class C(T:! type) {
  extend base: X(T);
}

fn F() {
  let i: i32 = C(i32).G();
}


// --- import_extend_generic_symbolic_base.carbon

library "[[@TEST_NAME]]";

import library "extend_generic_symbolic_base";

fn H() {
  let j: i32 = C(i32).G();
}

// CHECK:STDOUT: --- extend_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [template]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [template]
// CHECK:STDOUT:   %Base.0cc8: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.db1e: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.bb6d: type = unbound_element_type %Base.0cc8, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.710c: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60ab: <witness> = complete_type_witness %struct_type.x.710c [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [template]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [template]
// CHECK:STDOUT:   %complete_type.5e2b: <witness> = complete_type_witness %struct_type.y [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base.cc49: type = class_type @Base, @Base(%Param) [template]
// CHECK:STDOUT:   %Base.elem.8c24: type = unbound_element_type %Base.cc49, %Param [template]
// CHECK:STDOUT:   %struct_type.x.975c: type = struct_type {.x: %Param} [template]
// CHECK:STDOUT:   %complete_type.db3f: <witness> = complete_type_witness %struct_type.x.975c [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.cc49 [template]
// CHECK:STDOUT:   %struct_type.base.d475: type = struct_type {.base: %Base.cc49} [template]
// CHECK:STDOUT:   %complete_type.08e5: <witness> = complete_type_witness %struct_type.base.d475 [template]
// CHECK:STDOUT:   %DoubleFieldAccess.type: type = fn_type @DoubleFieldAccess [template]
// CHECK:STDOUT:   %DoubleFieldAccess: %DoubleFieldAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .DoubleFieldAccess = %DoubleFieldAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [template = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt.loc4_17.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_17.1, runtime_param<invalid> [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_17.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [template = constants.%Param] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [template = constants.%Derived] {} {}
// CHECK:STDOUT:   %DoubleFieldAccess.decl: %DoubleFieldAccess.type = fn_decl @DoubleFieldAccess [template = constants.%DoubleFieldAccess] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [template = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.1: type) {
// CHECK:STDOUT:   %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_17.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_17.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T.loc4_17.2 (%T) [symbolic = %require_complete (constants.%require_complete.db1e)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.0cc8)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.0cc8), @Base.%T.loc4_17.2 (%T) [symbolic = %Base.elem (constants.%Base.elem.bb6d)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710c)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.x (%struct_type.x.710c) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.bb6d) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.710c [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.60ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.0cc8
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl y, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.y [template = constants.%complete_type.5e2b]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .y = %.loc9
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [template = constants.%Base.generic]
// CHECK:STDOUT:   %Param.ref: type = name_ref Param, file.%Param.decl [template = constants.%Param]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [template = constants.%Base.cc49]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d475 [template = constants.%complete_type.08e5]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoubleFieldAccess(%d.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.8c24 = name_ref x, @Base.%.loc5 [template = @Base.%.loc5]
// CHECK:STDOUT:   %.loc17_11.1: ref %Base.cc49 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc17_11.2: ref %Base.cc49 = converted %d.ref, %.loc17_11.1
// CHECK:STDOUT:   %.loc17_11.3: ref %Param = class_element_access %.loc17_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, @Param.%.loc9 [template = @Param.%.loc9]
// CHECK:STDOUT:   %.loc17_13.1: ref %i32 = class_element_access %.loc17_11.3, element0
// CHECK:STDOUT:   %.loc17_13.2: %i32 = bind_value %.loc17_13.1
// CHECK:STDOUT:   return %.loc17_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T.loc4_17.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_17.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc4_17.2 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5e2b
// CHECK:STDOUT:   %Base => constants.%Base.cc49
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.8c24
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.975c
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.db3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [template]
// CHECK:STDOUT:   %complete_type.a2b5: <witness> = complete_type_witness %struct_type.y [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.710c: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.60ab: <witness> = complete_type_witness %struct_type.x.710c [symbolic]
// CHECK:STDOUT:   %Base.5e4b: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.802a: type = class_type @Base, @Base(%Param) [template]
// CHECK:STDOUT:   %struct_type.base.875c: type = struct_type {.base: %Base.802a} [template]
// CHECK:STDOUT:   %complete_type.8683: <witness> = complete_type_witness %struct_type.base.875c [template]
// CHECK:STDOUT:   %require_complete.db1e: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.acef: type = unbound_element_type %Base.5e4b, %T [symbolic]
// CHECK:STDOUT:   %Base.elem.8154: type = unbound_element_type %Base.802a, %Param [template]
// CHECK:STDOUT:   %struct_type.x.975c: type = struct_type {.x: %Param} [template]
// CHECK:STDOUT:   %complete_type.db3f: <witness> = complete_type_witness %struct_type.x.975c [template]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.type: type = fn_type @ImportedDoubleFieldAccess [template]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess: %ImportedDoubleFieldAccess.type = struct_value () [template]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.2108 = import_ref Main//extend_generic_base, Base, unloaded
// CHECK:STDOUT:   %import_ref.8772 = import_ref Main//extend_generic_base, Param, unloaded
// CHECK:STDOUT:   %import_ref.5024: type = import_ref Main//extend_generic_base, Derived, loaded [template = constants.%Derived]
// CHECK:STDOUT:   %import_ref.ac82 = import_ref Main//extend_generic_base, DoubleFieldAccess, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d695
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.f46b: <witness> = import_ref Main//extend_generic_base, loc10_1, loaded [template = constants.%complete_type.a2b5]
// CHECK:STDOUT:   %import_ref.4462 = import_ref Main//extend_generic_base, inst42 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c4e2: %Param.elem = import_ref Main//extend_generic_base, loc9_8, loaded [template = %.7d96]
// CHECK:STDOUT:   %import_ref.3eec: <witness> = import_ref Main//extend_generic_base, loc6_1, loaded [symbolic = @Base.%complete_type (constants.%complete_type.60ab)]
// CHECK:STDOUT:   %import_ref.a6b9 = import_ref Main//extend_generic_base, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8be7: @Base.%Base.elem (%Base.elem.acef) = import_ref Main//extend_generic_base, loc5_8, loaded [template = %.9ad0]
// CHECK:STDOUT:   %import_ref.6968: <witness> = import_ref Main//extend_generic_base, loc14_1, loaded [template = constants.%complete_type.8683]
// CHECK:STDOUT:   %import_ref.f6ca = import_ref Main//extend_generic_base, inst76 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.6262 = import_ref Main//extend_generic_base, loc13_27, unloaded
// CHECK:STDOUT:   %import_ref.6c6788.2: type = import_ref Main//extend_generic_base, loc13_26, loaded [template = constants.%Base.802a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = imports.%import_ref.2108
// CHECK:STDOUT:     .Param = imports.%import_ref.8772
// CHECK:STDOUT:     .Derived = imports.%import_ref.5024
// CHECK:STDOUT:     .DoubleFieldAccess = imports.%import_ref.ac82
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .ImportedDoubleFieldAccess = %ImportedDoubleFieldAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.decl: %ImportedDoubleFieldAccess.type = fn_decl @ImportedDoubleFieldAccess [template = constants.%ImportedDoubleFieldAccess] {
// CHECK:STDOUT:     %d.patt: %Derived = binding_pattern d
// CHECK:STDOUT:     %d.param_patt: %Derived = value_param_pattern %d.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param runtime_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, imports.%import_ref.5024 [template = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived [from "extend_generic_base.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f6ca
// CHECK:STDOUT:   .base = imports.%import_ref.6262
// CHECK:STDOUT:   extend imports.%import_ref.6c6788.2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.6968
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param [from "extend_generic_base.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4462
// CHECK:STDOUT:   .y = imports.%import_ref.c4e2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.f46b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(constants.%T: type) [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T (%T) [symbolic = %require_complete (constants.%require_complete.db1e)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.5e4b)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.5e4b), @Base.%T (%T) [symbolic = %Base.elem (constants.%Base.elem.acef)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.710c)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Base.%struct_type.x (%struct_type.x.710c) [symbolic = %complete_type (constants.%complete_type.60ab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.a6b9
// CHECK:STDOUT:     .x = imports.%import_ref.8be7
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3eec
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedDoubleFieldAccess(%d.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.8154 = name_ref x, imports.%import_ref.8be7 [template = imports.%.9ad0]
// CHECK:STDOUT:   %.loc7_11.1: ref %Base.802a = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc7_11.2: ref %Base.802a = converted %d.ref, %.loc7_11.1
// CHECK:STDOUT:   %.loc7_11.3: ref %Param = class_element_access %.loc7_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, imports.%import_ref.c4e2 [template = imports.%.7d96]
// CHECK:STDOUT:   %.loc7_13.1: ref %i32 = class_element_access %.loc7_11.3, element0
// CHECK:STDOUT:   %.loc7_13.2: %i32 = bind_value %.loc7_13.1
// CHECK:STDOUT:   return %.loc7_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %T.patt => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.a2b5
// CHECK:STDOUT:   %Base => constants.%Base.802a
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.8154
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.975c
// CHECK:STDOUT:   %complete_type => constants.%complete_type.db3f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.f9e5: type = class_type @C, @C(%X) [template]
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
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [template = constants.%X] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%T.loc4_9.2 (%T) [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.1 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %.loc8: <error> = base_decl <error>, element0 [template]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [template = <error>]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e2
// CHECK:STDOUT:     .base = %.loc8
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .G = %G.decl
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [template = constants.%X]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%X) [template = constants.%C.f9e5]
// CHECK:STDOUT:   %G.ref: <error> = name_ref G, <error> [template = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%X
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [template]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [template]
// CHECK:STDOUT:   %X.8471: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.type.9e73: type = fn_type @G, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.ca3a: %G.type.9e73 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.4278: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %G.specific_fn.a95b: <specific function> = specific_function %G.ca3a, @G(%U) [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.37e2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %X.643b: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.type.709b: type = fn_type @G, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.0f4b: %G.type.709b = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.c569: <witness> = require_complete_type %X.643b [symbolic]
// CHECK:STDOUT:   %C.elem.5136: type = unbound_element_type %C.37e2, %X.643b [symbolic]
// CHECK:STDOUT:   %struct_type.base.8ef5: type = struct_type {.base: %X.643b} [symbolic]
// CHECK:STDOUT:   %complete_type.bbcb: <witness> = complete_type_witness %struct_type.base.8ef5 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a3: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.2685: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %X.05fc: type = class_type @X, @X(%i32) [template]
// CHECK:STDOUT:   %G.type.d5ee: type = fn_type @G, @X(%i32) [template]
// CHECK:STDOUT:   %G.8429: %G.type.d5ee = struct_value () [template]
// CHECK:STDOUT:   %C.elem.6108: type = unbound_element_type %C.2685, %X.05fc [template]
// CHECK:STDOUT:   %struct_type.base.00d1: type = struct_type {.base: %X.05fc} [template]
// CHECK:STDOUT:   %complete_type.b698: <witness> = complete_type_witness %struct_type.base.00d1 [template]
// CHECK:STDOUT:   %G.specific_fn.af5c: <specific function> = specific_function %G.8429, @G(%i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [template = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt.loc4_14.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_14.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc4_14.1, runtime_param<invalid> [symbolic = %U.patt.loc4_14.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc4_14.1: type = bind_symbolic_name U, 0, %U.param [symbolic = %U.loc4_14.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt.loc8_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc8_9.1, runtime_param<invalid> [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc8_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc4_14.1: type) {
// CHECK:STDOUT:   %U.loc4_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_14.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc4_14.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc4_14.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U.loc4_14.2) [symbolic = %G.type (constants.%G.type.9e73)]
// CHECK:STDOUT:   %G: @X.%G.type (%G.type.9e73) = struct_value () [symbolic = %G (constants.%G.ca3a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %G.decl: @X.%G.type (%G.type.9e73) = fn_decl @G [symbolic = @X.%G (constants.%G.ca3a)] {
// CHECK:STDOUT:       %return.patt: @G.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%U (%U) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc4_14.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @G.%U (%U) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @G.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3571]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X.8471
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.1: type) {
// CHECK:STDOUT:   %T.loc8_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc8_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2: type = class_type @X, @X(%T.loc8_9.2) [symbolic = %X.loc9_19.2 (constants.%X.643b)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%X.loc9_19.2 (%X.643b) [symbolic = %require_complete (constants.%require_complete.c569)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc8_9.2) [symbolic = %C (constants.%C.37e2)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.37e2), @C.%X.loc9_19.2 (%X.643b) [symbolic = %C.elem (constants.%C.elem.5136)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X.loc9_19.2 (%X.643b)} [symbolic = %struct_type.base (constants.%struct_type.base.8ef5)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness @C.%struct_type.base (%struct_type.base.8ef5) [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bbcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.ref: %X.type = name_ref X, file.%X.decl [template = constants.%X.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_9.1 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc9_19.1: type = class_type @X, @X(constants.%T) [symbolic = %X.loc9_19.2 (constants.%X.643b)]
// CHECK:STDOUT:     %.loc9: @C.%C.elem (%C.elem.5136) = base_decl %X.loc9_19.1, element0 [template]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness %struct_type.base.8ef5 [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bbcb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.37e2
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     extend %X.loc9_19.1
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@X.%U.loc4_14.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%U (%U) [symbolic = %require_complete (constants.%require_complete.4278)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.9e73)]
// CHECK:STDOUT:   %G: @G.%G.type (%G.type.9e73) = struct_value () [symbolic = %G (constants.%G.ca3a)]
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2: <specific function> = specific_function %G, @G(%U) [symbolic = %G.specific_fn.loc5_24.2 (constants.%G.specific_fn.a95b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @G.%U (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_24: @G.%G.type (%G.type.9e73) = specific_constant @X.%G.decl, @X(constants.%U) [symbolic = %G (constants.%G.ca3a)]
// CHECK:STDOUT:     %G.ref: @G.%G.type (%G.type.9e73) = name_ref G, %.loc5_24 [symbolic = %G (constants.%G.ca3a)]
// CHECK:STDOUT:     %G.specific_fn.loc5_24.1: <specific function> = specific_function %G.ref, @G(constants.%U) [symbolic = %G.specific_fn.loc5_24.2 (constants.%G.specific_fn.a95b)]
// CHECK:STDOUT:     %G.call: init @G.%U (%U) = call %G.specific_fn.loc5_24.1()
// CHECK:STDOUT:     %.loc5_27.1: @G.%U (%U) = value_of_initializer %G.call
// CHECK:STDOUT:     %.loc5_27.2: @G.%U (%U) = converted %G.call, %.loc5_27.1
// CHECK:STDOUT:     return %.loc5_27.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2685]
// CHECK:STDOUT:   %.loc13_22: %G.type.d5ee = specific_constant @X.%G.decl, @X(constants.%i32) [template = constants.%G.8429]
// CHECK:STDOUT:   %G.ref: %G.type.d5ee = name_ref G, %.loc13_22 [template = constants.%G.8429]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [template = constants.%G.specific_fn.af5c]
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn()
// CHECK:STDOUT:   %.loc13_26.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc13_26.2: %i32 = converted %G.call, %.loc13_26.1
// CHECK:STDOUT:   %i: %i32 = bind_name i, %.loc13_26.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc4_14.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc4_14.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.9e73
// CHECK:STDOUT:   %G => constants.%G.ca3a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4278
// CHECK:STDOUT:   %G.type => constants.%G.type.9e73
// CHECK:STDOUT:   %G => constants.%G.ca3a
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2 => constants.%G.specific_fn.a95b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(%U.loc4_14.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@G.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U.loc4_14.2 => constants.%T
// CHECK:STDOUT:   %U.patt.loc4_14.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.709b
// CHECK:STDOUT:   %G => constants.%G.0f4b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@C.%T.loc8_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T.loc8_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_9.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc8_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2 => constants.%X.05fc
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3571
// CHECK:STDOUT:   %C => constants.%C.2685
// CHECK:STDOUT:   %C.elem => constants.%C.elem.6108
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.00d1
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.b698
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U.loc4_14.2 => constants.%i32
// CHECK:STDOUT:   %U.patt.loc4_14.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.d5ee
// CHECK:STDOUT:   %G => constants.%G.8429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a3
// CHECK:STDOUT:   %G.type => constants.%G.type.d5ee
// CHECK:STDOUT:   %G => constants.%G.8429
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2 => constants.%G.specific_fn.af5c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %H.type: type = fn_type @H [template]
// CHECK:STDOUT:   %H: %H.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f8a3: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3571: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.ab1a: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %C.df41: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.b012: type = unbound_element_type %C.df41, %X.ab1a [symbolic]
// CHECK:STDOUT:   %struct_type.base.4044: type = struct_type {.base: %X.ab1a} [symbolic]
// CHECK:STDOUT:   %complete_type.fe76: <witness> = complete_type_witness %struct_type.base.4044 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type.804e: type = fn_type @G, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.ed3d: %G.type.804e = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.8e81: <witness> = require_complete_type %X.ab1a [symbolic]
// CHECK:STDOUT:   %require_complete.4278: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %G.specific_fn.3dd2: <specific function> = specific_function %G.ed3d, @G(%U) [symbolic]
// CHECK:STDOUT:   %G.type.aaf4: type = fn_type @G, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.ec7e: %G.type.aaf4 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.733f: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %X.3bf3: type = class_type @X, @X(%i32) [template]
// CHECK:STDOUT:   %G.type.aacc: type = fn_type @G, @X(%i32) [template]
// CHECK:STDOUT:   %G.08fa: %G.type.aacc = struct_value () [template]
// CHECK:STDOUT:   %C.elem.1234: type = unbound_element_type %C.733f, %X.3bf3 [template]
// CHECK:STDOUT:   %struct_type.base.1d9c: type = struct_type {.base: %X.3bf3} [template]
// CHECK:STDOUT:   %complete_type.82bb: <witness> = complete_type_witness %struct_type.base.1d9c [template]
// CHECK:STDOUT:   %G.specific_fn.e4a7: <specific function> = specific_function %G.08fa, @G(%i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.4de4 = import_ref Main//extend_generic_symbolic_base, X, unloaded
// CHECK:STDOUT:   %import_ref.26e8: %C.type = import_ref Main//extend_generic_symbolic_base, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.b07f = import_ref Main//extend_generic_symbolic_base, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.187e
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f24: <witness> = import_ref Main//extend_generic_symbolic_base, loc6_1, loaded [template = constants.%complete_type.3571]
// CHECK:STDOUT:   %import_ref.8391 = import_ref Main//extend_generic_symbolic_base, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ba88: @X.%G.type (%G.type.804e) = import_ref Main//extend_generic_symbolic_base, loc5_15, loaded [symbolic = @X.%G (constants.%G.ed3d)]
// CHECK:STDOUT:   %import_ref.bb26: <witness> = import_ref Main//extend_generic_symbolic_base, loc10_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.fe76)]
// CHECK:STDOUT:   %import_ref.495d = import_ref Main//extend_generic_symbolic_base, inst68 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.004d = import_ref Main//extend_generic_symbolic_base, loc9_20, unloaded
// CHECK:STDOUT:   %import_ref.b59216.2: type = import_ref Main//extend_generic_symbolic_base, loc9_19, loaded [symbolic = @C.%X (constants.%X.ab1a)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .X = imports.%import_ref.4de4
// CHECK:STDOUT:     .C = imports.%import_ref.26e8
// CHECK:STDOUT:     .F = imports.%import_ref.b07f
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <invalid>
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [template = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(constants.%T: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X: type = class_type @X, @X(%T) [symbolic = %X (constants.%X.ab1a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%X (%X.ab1a) [symbolic = %require_complete (constants.%require_complete.8e81)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.df41)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.df41), @C.%X (%X.ab1a) [symbolic = %C.elem (constants.%C.elem.b012)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X (%X.ab1a)} [symbolic = %struct_type.base (constants.%struct_type.base.4044)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.base (%struct_type.base.4044) [symbolic = %complete_type (constants.%complete_type.fe76)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.495d
// CHECK:STDOUT:     .base = imports.%import_ref.004d
// CHECK:STDOUT:     extend imports.%import_ref.b59216.2
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.bb26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(constants.%U: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.804e)]
// CHECK:STDOUT:   %G: @X.%G.type (%G.type.804e) = struct_value () [symbolic = %G (constants.%G.ed3d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.8391
// CHECK:STDOUT:     .G = imports.%import_ref.ba88
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f24
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.26e8 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.733f]
// CHECK:STDOUT:   %.loc7_22: %G.type.aacc = specific_constant imports.%import_ref.ba88, @X(constants.%i32) [template = constants.%G.08fa]
// CHECK:STDOUT:   %G.ref: %G.type.aacc = name_ref G, %.loc7_22 [template = constants.%G.08fa]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [template = constants.%G.specific_fn.e4a7]
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn()
// CHECK:STDOUT:   %.loc7_26.1: %i32 = value_of_initializer %G.call
// CHECK:STDOUT:   %.loc7_26.2: %i32 = converted %G.call, %.loc7_26.1
// CHECK:STDOUT:   %j: %i32 = bind_name j, %.loc7_26.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(constants.%U: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%U (%U) [symbolic = %require_complete (constants.%require_complete.4278)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.804e)]
// CHECK:STDOUT:   %G: @G.%G.type (%G.type.804e) = struct_value () [symbolic = %G (constants.%G.ed3d)]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%U) [symbolic = %G.specific_fn (constants.%G.specific_fn.3dd2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @G.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.804e
// CHECK:STDOUT:   %G => constants.%G.ed3d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U => constants.%T
// CHECK:STDOUT:   %U.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.aaf4
// CHECK:STDOUT:   %G => constants.%G.ec7e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@C.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4278
// CHECK:STDOUT:   %G.type => constants.%G.type.804e
// CHECK:STDOUT:   %G => constants.%G.ed3d
// CHECK:STDOUT:   %G.specific_fn => constants.%G.specific_fn.3dd2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@G.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:   %T.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X => constants.%X.3bf3
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3571
// CHECK:STDOUT:   %C => constants.%C.733f
// CHECK:STDOUT:   %C.elem => constants.%C.elem.1234
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.1d9c
// CHECK:STDOUT:   %complete_type => constants.%complete_type.82bb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %U.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.aacc
// CHECK:STDOUT:   %G => constants.%G.08fa
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a3
// CHECK:STDOUT:   %G.type => constants.%G.type.aacc
// CHECK:STDOUT:   %G => constants.%G.08fa
// CHECK:STDOUT:   %G.specific_fn => constants.%G.specific_fn.e4a7
// CHECK:STDOUT: }
// CHECK:STDOUT:
