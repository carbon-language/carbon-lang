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
// CHECK:STDOUT:   %Base.0: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.d: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.b: type = unbound_element_type %Base.0, %T [symbolic]
// CHECK:STDOUT:   %struct_type.x.7: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.6: <witness> = complete_type_witness %struct_type.x.7 [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [template]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [template]
// CHECK:STDOUT:   %complete_type.5: <witness> = complete_type_witness %struct_type.y [template]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [template]
// CHECK:STDOUT:   %Base.c: type = class_type @Base, @Base(%Param) [template]
// CHECK:STDOUT:   %Base.elem.8: type = unbound_element_type %Base.c, %Param [template]
// CHECK:STDOUT:   %struct_type.x.9: type = struct_type {.x: %Param} [template]
// CHECK:STDOUT:   %complete_type.d: <witness> = complete_type_witness %struct_type.x.9 [template]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.c [template]
// CHECK:STDOUT:   %struct_type.base.d: type = struct_type {.base: %Base.c} [template]
// CHECK:STDOUT:   %complete_type.0: <witness> = complete_type_witness %struct_type.base.d [template]
// CHECK:STDOUT:   %DoubleFieldAccess.type: type = fn_type @DoubleFieldAccess [template]
// CHECK:STDOUT:   %DoubleFieldAccess: %DoubleFieldAccess.type = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T.loc4_17.2 (%T) [symbolic = %require_complete (constants.%require_complete.d)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.2) [symbolic = %Base (constants.%Base.0)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.0), @Base.%T.loc4_17.2 (%T) [symbolic = %Base.elem (constants.%Base.elem.b)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T.loc4_17.2 (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.7)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness @Base.%struct_type.x (%struct_type.x.7) [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.b) = field_decl x, element0 [template]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness %struct_type.x.7 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.0
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl y, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.y [template = constants.%complete_type.5]
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
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [template = constants.%Base.c]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %Base, element0 [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base.d [template = constants.%complete_type.0]
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
// CHECK:STDOUT:   %x.ref: %Base.elem.8 = name_ref x, @Base.%.loc5 [template = @Base.%.loc5]
// CHECK:STDOUT:   %.loc17_11.1: ref %Base.c = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc17_11.2: ref %Base.c = converted %d.ref, %.loc17_11.1
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.5
// CHECK:STDOUT:   %Base => constants.%Base.c
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.9
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.d
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
// CHECK:STDOUT:   %complete_type.a: <witness> = complete_type_witness %struct_type.y [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.7: type = struct_type {.x: %T} [symbolic]
// CHECK:STDOUT:   %complete_type.6: <witness> = complete_type_witness %struct_type.x.7 [symbolic]
// CHECK:STDOUT:   %Base.5: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Base.8: type = class_type @Base, @Base(%Param) [template]
// CHECK:STDOUT:   %struct_type.base.8: type = struct_type {.base: %Base.8} [template]
// CHECK:STDOUT:   %complete_type.8: <witness> = complete_type_witness %struct_type.base.8 [template]
// CHECK:STDOUT:   %require_complete.d: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Base.elem.a: type = unbound_element_type %Base.5, %T [symbolic]
// CHECK:STDOUT:   %Base.elem.8: type = unbound_element_type %Base.8, %Param [template]
// CHECK:STDOUT:   %struct_type.x.9: type = struct_type {.x: %Param} [template]
// CHECK:STDOUT:   %complete_type.d: <witness> = complete_type_witness %struct_type.x.9 [template]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.type: type = fn_type @ImportedDoubleFieldAccess [template]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess: %ImportedDoubleFieldAccess.type = struct_value () [template]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.2 = import_ref Main//extend_generic_base, Base, unloaded
// CHECK:STDOUT:   %import_ref.87 = import_ref Main//extend_generic_base, Param, unloaded
// CHECK:STDOUT:   %import_ref.5: type = import_ref Main//extend_generic_base, Derived, loaded [template = constants.%Derived]
// CHECK:STDOUT:   %import_ref.ac = import_ref Main//extend_generic_base, DoubleFieldAccess, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.d
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.f4: <witness> = import_ref Main//extend_generic_base, loc10_1, loaded [template = constants.%complete_type.a]
// CHECK:STDOUT:   %import_ref.4 = import_ref Main//extend_generic_base, inst42 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.c4: %Param.elem = import_ref Main//extend_generic_base, loc9_8, loaded [template = %.7]
// CHECK:STDOUT:   %import_ref.3: <witness> = import_ref Main//extend_generic_base, loc6_1, loaded [symbolic = @Base.%complete_type (constants.%complete_type.6)]
// CHECK:STDOUT:   %import_ref.a6 = import_ref Main//extend_generic_base, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.8b: @Base.%Base.elem (%Base.elem.a) = import_ref Main//extend_generic_base, loc5_8, loaded [template = %.9]
// CHECK:STDOUT:   %import_ref.69: <witness> = import_ref Main//extend_generic_base, loc14_1, loaded [template = constants.%complete_type.8]
// CHECK:STDOUT:   %import_ref.f6 = import_ref Main//extend_generic_base, inst76 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.62 = import_ref Main//extend_generic_base, loc13_27, unloaded
// CHECK:STDOUT:   %import_ref.6c6788.2: type = import_ref Main//extend_generic_base, loc13_26, loaded [template = constants.%Base.8]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Base = imports.%import_ref.2
// CHECK:STDOUT:     .Param = imports.%import_ref.87
// CHECK:STDOUT:     .Derived = imports.%import_ref.5
// CHECK:STDOUT:     .DoubleFieldAccess = imports.%import_ref.ac
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
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, imports.%import_ref.5 [template = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = bind_name d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived [from "extend_generic_base.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.f6
// CHECK:STDOUT:   .base = imports.%import_ref.62
// CHECK:STDOUT:   extend imports.%import_ref.6c6788.2
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param [from "extend_generic_base.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%import_ref.4
// CHECK:STDOUT:   .y = imports.%import_ref.c4
// CHECK:STDOUT:   complete_type_witness = imports.%import_ref.f4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(constants.%T: type) [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Base.%T (%T) [symbolic = %require_complete (constants.%require_complete.d)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.5)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type @Base.%Base (%Base.5), @Base.%T (%T) [symbolic = %Base.elem (constants.%Base.elem.a)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T (%T)} [symbolic = %struct_type.x (constants.%struct_type.x.7)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @Base.%struct_type.x (%struct_type.x.7) [symbolic = %complete_type (constants.%complete_type.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.a6
// CHECK:STDOUT:     .x = imports.%import_ref.8b
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.3
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedDoubleFieldAccess(%d.param_patt: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.8 = name_ref x, imports.%import_ref.8b [template = imports.%.9]
// CHECK:STDOUT:   %.loc7_11.1: ref %Base.8 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc7_11.2: ref %Base.8 = converted %d.ref, %.loc7_11.1
// CHECK:STDOUT:   %.loc7_11.3: ref %Param = class_element_access %.loc7_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, imports.%import_ref.c4 [template = imports.%.7]
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
// CHECK:STDOUT:   %require_complete => constants.%complete_type.a
// CHECK:STDOUT:   %Base => constants.%Base.8
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.9
// CHECK:STDOUT:   %complete_type => constants.%complete_type.d
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
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.f: type = class_type @C, @C(%X) [template]
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
// CHECK:STDOUT:     .Self = constants.%C.3
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%X) [template = constants.%C.f]
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
// CHECK:STDOUT:   %X.8: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.type.9: type = fn_type @G, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.c: %G.type.9 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %G.specific_fn.a9: <specific function> = specific_function %G.c, @G(%U) [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %X.6: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.type.7: type = fn_type @G, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.0: %G.type.7 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.c: <witness> = require_complete_type %X.6 [symbolic]
// CHECK:STDOUT:   %C.elem.5: type = unbound_element_type %C.3, %X.6 [symbolic]
// CHECK:STDOUT:   %struct_type.base.8: type = struct_type {.base: %X.6} [symbolic]
// CHECK:STDOUT:   %complete_type.bb: <witness> = complete_type_witness %struct_type.base.8 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [template]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [template]
// CHECK:STDOUT:   %complete_type.f: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %X.0: type = class_type @X, @X(%i32) [template]
// CHECK:STDOUT:   %G.type.d: type = fn_type @G, @X(%i32) [template]
// CHECK:STDOUT:   %G.8: %G.type.d = struct_value () [template]
// CHECK:STDOUT:   %C.elem.6: type = unbound_element_type %C.2, %X.0 [template]
// CHECK:STDOUT:   %struct_type.base.0: type = struct_type {.base: %X.0} [template]
// CHECK:STDOUT:   %complete_type.b6: <witness> = complete_type_witness %struct_type.base.0 [template]
// CHECK:STDOUT:   %G.specific_fn.af: <specific function> = specific_function %G.8, @G(%i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
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
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U.loc4_14.2) [symbolic = %G.type (constants.%G.type.9)]
// CHECK:STDOUT:   %G: @X.%G.type (%G.type.9) = struct_value () [symbolic = %G (constants.%G.c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %G.decl: @X.%G.type (%G.type.9) = fn_decl @G [symbolic = @X.%G (constants.%G.c)] {
// CHECK:STDOUT:       %return.patt: @G.%U (%U) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%U (%U) = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc4_14.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @G.%U (%U) = out_param runtime_param0
// CHECK:STDOUT:       %return: ref @G.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [template = constants.%complete_type.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X.8
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
// CHECK:STDOUT:   %X.loc9_19.2: type = class_type @X, @X(%T.loc8_9.2) [symbolic = %X.loc9_19.2 (constants.%X.6)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%X.loc9_19.2 (%X.6) [symbolic = %require_complete (constants.%require_complete.c)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc8_9.2) [symbolic = %C (constants.%C.3)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.3), @C.%X.loc9_19.2 (%X.6) [symbolic = %C.elem (constants.%C.elem.5)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X.loc9_19.2 (%X.6)} [symbolic = %struct_type.base (constants.%struct_type.base.8)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness @C.%struct_type.base (%struct_type.base.8) [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.ref: %X.type = name_ref X, file.%X.decl [template = constants.%X.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_9.1 [symbolic = %T.loc8_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc9_19.1: type = class_type @X, @X(constants.%T) [symbolic = %X.loc9_19.2 (constants.%X.6)]
// CHECK:STDOUT:     %.loc9: @C.%C.elem (%C.elem.5) = base_decl %X.loc9_19.1, element0 [template]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness %struct_type.base.8 [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%U (%U) [symbolic = %require_complete (constants.%require_complete.4)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.9)]
// CHECK:STDOUT:   %G: @G.%G.type (%G.type.9) = struct_value () [symbolic = %G (constants.%G.c)]
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2: <specific function> = specific_function %G, @G(%U) [symbolic = %G.specific_fn.loc5_24.2 (constants.%G.specific_fn.a9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @G.%U (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_24: @G.%G.type (%G.type.9) = specific_constant @X.%G.decl, @X(constants.%U) [symbolic = %G (constants.%G.c)]
// CHECK:STDOUT:     %G.ref: @G.%G.type (%G.type.9) = name_ref G, %.loc5_24 [symbolic = %G (constants.%G.c)]
// CHECK:STDOUT:     %G.specific_fn.loc5_24.1: <specific function> = specific_function %G.ref, @G(constants.%U) [symbolic = %G.specific_fn.loc5_24.2 (constants.%G.specific_fn.a9)]
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.2]
// CHECK:STDOUT:   %.loc13_22: %G.type.d = specific_constant @X.%G.decl, @X(constants.%i32) [template = constants.%G.8]
// CHECK:STDOUT:   %G.ref: %G.type.d = name_ref G, %.loc13_22 [template = constants.%G.8]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [template = constants.%G.specific_fn.af]
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
// CHECK:STDOUT:   %G.type => constants.%G.type.9
// CHECK:STDOUT:   %G => constants.%G.c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4
// CHECK:STDOUT:   %G.type => constants.%G.type.9
// CHECK:STDOUT:   %G => constants.%G.c
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2 => constants.%G.specific_fn.a9
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
// CHECK:STDOUT:   %G.type => constants.%G.type.7
// CHECK:STDOUT:   %G => constants.%G.0
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
// CHECK:STDOUT:   %X.loc9_19.2 => constants.%X.0
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %C => constants.%C.2
// CHECK:STDOUT:   %C.elem => constants.%C.elem.6
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.0
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U.loc4_14.2 => constants.%i32
// CHECK:STDOUT:   %U.patt.loc4_14.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.d
// CHECK:STDOUT:   %G => constants.%G.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f
// CHECK:STDOUT:   %G.type => constants.%G.type.d
// CHECK:STDOUT:   %G => constants.%G.8
// CHECK:STDOUT:   %G.specific_fn.loc5_24.2 => constants.%G.specific_fn.af
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
// CHECK:STDOUT:   %complete_type.f8: <witness> = complete_type_witness %i32.builtin [template]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [template]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [template]
// CHECK:STDOUT:   %complete_type.3: <witness> = complete_type_witness %empty_struct_type [template]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.ab: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %C.d: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.b: type = unbound_element_type %C.d, %X.ab [symbolic]
// CHECK:STDOUT:   %struct_type.base.4: type = struct_type {.base: %X.ab} [symbolic]
// CHECK:STDOUT:   %complete_type.fe: <witness> = complete_type_witness %struct_type.base.4 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %G.type.8: type = fn_type @G, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.ed: %G.type.8 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.8e: <witness> = require_complete_type %X.ab [symbolic]
// CHECK:STDOUT:   %require_complete.4: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %G.specific_fn.3: <specific function> = specific_function %G.ed, @G(%U) [symbolic]
// CHECK:STDOUT:   %G.type.aaf: type = fn_type @G, @X(%T) [symbolic]
// CHECK:STDOUT:   %G.ec: %G.type.aaf = struct_value () [symbolic]
// CHECK:STDOUT:   %C.7: type = class_type @C, @C(%i32) [template]
// CHECK:STDOUT:   %X.3: type = class_type @X, @X(%i32) [template]
// CHECK:STDOUT:   %G.type.aac: type = fn_type @G, @X(%i32) [template]
// CHECK:STDOUT:   %G.0: %G.type.aac = struct_value () [template]
// CHECK:STDOUT:   %C.elem.1: type = unbound_element_type %C.7, %X.3 [template]
// CHECK:STDOUT:   %struct_type.base.1: type = struct_type {.base: %X.3} [template]
// CHECK:STDOUT:   %complete_type.8: <witness> = complete_type_witness %struct_type.base.1 [template]
// CHECK:STDOUT:   %G.specific_fn.e: <specific function> = specific_function %G.0, @G(%i32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %import_ref.4d = import_ref Main//extend_generic_symbolic_base, X, unloaded
// CHECK:STDOUT:   %import_ref.2: %C.type = import_ref Main//extend_generic_symbolic_base, C, loaded [template = constants.%C.generic]
// CHECK:STDOUT:   %import_ref.b0 = import_ref Main//extend_generic_symbolic_base, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref.8f: <witness> = import_ref Main//extend_generic_symbolic_base, loc6_1, loaded [template = constants.%complete_type.3]
// CHECK:STDOUT:   %import_ref.83 = import_ref Main//extend_generic_symbolic_base, inst27 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.ba: @X.%G.type (%G.type.8) = import_ref Main//extend_generic_symbolic_base, loc5_15, loaded [symbolic = @X.%G (constants.%G.ed)]
// CHECK:STDOUT:   %import_ref.bb: <witness> = import_ref Main//extend_generic_symbolic_base, loc10_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.fe)]
// CHECK:STDOUT:   %import_ref.49 = import_ref Main//extend_generic_symbolic_base, inst68 [no loc], unloaded
// CHECK:STDOUT:   %import_ref.00 = import_ref Main//extend_generic_symbolic_base, loc9_20, unloaded
// CHECK:STDOUT:   %import_ref.b59216.2: type = import_ref Main//extend_generic_symbolic_base, loc9_19, loaded [symbolic = @C.%X (constants.%X.ab)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .X = imports.%import_ref.4d
// CHECK:STDOUT:     .C = imports.%import_ref.2
// CHECK:STDOUT:     .F = imports.%import_ref.b0
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
// CHECK:STDOUT:   %X: type = class_type @X, @X(%T) [symbolic = %X (constants.%X.ab)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @C.%X (%X.ab) [symbolic = %require_complete (constants.%require_complete.8e)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.d)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type @C.%C (%C.d), @C.%X (%X.ab) [symbolic = %C.elem (constants.%C.elem.b)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X (%X.ab)} [symbolic = %struct_type.base (constants.%struct_type.base.4)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness @C.%struct_type.base (%struct_type.base.4) [symbolic = %complete_type (constants.%complete_type.fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.49
// CHECK:STDOUT:     .base = imports.%import_ref.00
// CHECK:STDOUT:     extend imports.%import_ref.b59216.2
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.bb
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(constants.%U: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.8)]
// CHECK:STDOUT:   %G: @X.%G.type (%G.type.8) = struct_value () [symbolic = %G (constants.%G.ed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%import_ref.83
// CHECK:STDOUT:     .G = imports.%import_ref.ba
// CHECK:STDOUT:     complete_type_witness = imports.%import_ref.8f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%import_ref.2 [template = constants.%C.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [template = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [template = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [template = constants.%C.7]
// CHECK:STDOUT:   %.loc7_22: %G.type.aac = specific_constant imports.%import_ref.ba, @X(constants.%i32) [template = constants.%G.0]
// CHECK:STDOUT:   %G.ref: %G.type.aac = name_ref G, %.loc7_22 [template = constants.%G.0]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [template = constants.%G.specific_fn.e]
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
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%U (%U) [symbolic = %require_complete (constants.%require_complete.4)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @X(%U) [symbolic = %G.type (constants.%G.type.8)]
// CHECK:STDOUT:   %G: @G.%G.type (%G.type.8) = struct_value () [symbolic = %G (constants.%G.ed)]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%U) [symbolic = %G.specific_fn (constants.%G.specific_fn.3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @G.%U (%U);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.8
// CHECK:STDOUT:   %G => constants.%G.ed
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U => constants.%T
// CHECK:STDOUT:   %U.patt => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.aaf
// CHECK:STDOUT:   %G => constants.%G.ec
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
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4
// CHECK:STDOUT:   %G.type => constants.%G.type.8
// CHECK:STDOUT:   %G => constants.%G.ed
// CHECK:STDOUT:   %G.specific_fn => constants.%G.specific_fn.3
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
// CHECK:STDOUT:   %X => constants.%X.3
// CHECK:STDOUT:   %require_complete => constants.%complete_type.3
// CHECK:STDOUT:   %C => constants.%C.7
// CHECK:STDOUT:   %C.elem => constants.%C.elem.1
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.1
// CHECK:STDOUT:   %complete_type => constants.%complete_type.8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %U.patt => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %G.type => constants.%G.type.aac
// CHECK:STDOUT:   %G => constants.%G.0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8
// CHECK:STDOUT:   %G.type => constants.%G.type.aac
// CHECK:STDOUT:   %G => constants.%G.0
// CHECK:STDOUT:   %G.specific_fn => constants.%G.specific_fn.e
// CHECK:STDOUT: }
// CHECK:STDOUT:
