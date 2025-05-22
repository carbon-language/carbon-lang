// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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

//@dump-sem-ir-begin
class Derived {
  extend base: Base(Param);
}
//@dump-sem-ir-end

fn DoubleFieldAccess(d: Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.x.y;
  //@dump-sem-ir-end
}

// --- import.carbon

library "[[@TEST_NAME]]";

import library "extend_generic_base";

fn ImportedDoubleFieldAccess(d: Derived) -> i32 {
  //@dump-sem-ir-begin
  return d.x.y;
  //@dump-sem-ir-end
}

// --- fail_todo_extend_symbolic_base.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {
  // CHECK:STDERR: fail_todo_extend_symbolic_base.carbon:[[@LINE+4]]:16: error: deriving from final type `T`; base type must be an `abstract` or `base` class [BaseIsFinal]
  // CHECK:STDERR:   extend base: T;
  // CHECK:STDERR:                ^
  // CHECK:STDERR:
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

//@dump-sem-ir-begin
class C(T:! type) {
  extend base: X(T);
}
//@dump-sem-ir-end

fn F() {
  //@dump-sem-ir-begin
  let i: i32 = C(i32).G();
  //@dump-sem-ir-end
}


// --- import_extend_generic_symbolic_base.carbon

library "[[@TEST_NAME]]";

import library "extend_generic_symbolic_base";

fn H() {
  //@dump-sem-ir-begin
  let i: i32 = C(i32).G();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- extend_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.7a8: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %Base.elem.d1f: type = unbound_element_type %Base.7a8, %Param [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.7a8 [concrete]
// CHECK:STDOUT:   %struct_type.base.8bc: type = struct_type {.base: %Base.7a8} [concrete]
// CHECK:STDOUT:   %complete_type.b07: <witness> = complete_type_witness %struct_type.base.8bc [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [concrete = constants.%Base.7a8]
// CHECK:STDOUT:   %.loc14: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: %Base.7a8} [concrete = constants.%struct_type.base.8bc]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [concrete = constants.%complete_type.b07]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .base = %.loc14
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoubleFieldAccess(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.d1f = name_ref x, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc20_11.1: ref %Base.7a8 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc20_11.2: ref %Base.7a8 = converted %d.ref, %.loc20_11.1
// CHECK:STDOUT:   %.loc20_11.3: ref %Param = class_element_access %.loc20_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc20_13.1: ref %i32 = class_element_access %.loc20_11.3, element0
// CHECK:STDOUT:   %.loc20_13.2: %i32 = bind_value %.loc20_13.1
// CHECK:STDOUT:   return %.loc20_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T) [symbolic]
// CHECK:STDOUT:   %Base.7a8: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T [symbolic]
// CHECK:STDOUT:   %Base.elem.d1f: type = unbound_element_type %Base.7a8, %Param [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.import_ref.a92: %Param.elem = import_ref Main//extend_generic_base, loc9_8, loaded [concrete = %.be7]
// CHECK:STDOUT:   %Main.import_ref.7f7: @Base.%Base.elem (%Base.elem.9af) = import_ref Main//extend_generic_base, loc5_8, loaded [concrete = %.e66]
// CHECK:STDOUT:   %.e66: @Base.%Base.elem (%Base.elem.9af) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %.be7: %Param.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedDoubleFieldAccess(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.d1f = name_ref x, imports.%Main.import_ref.7f7 [concrete = imports.%.e66]
// CHECK:STDOUT:   %.loc8_11.1: ref %Base.7a8 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc8_11.2: ref %Base.7a8 = converted %d.ref, %.loc8_11.1
// CHECK:STDOUT:   %.loc8_11.3: ref %Param = class_element_access %.loc8_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, imports.%Main.import_ref.a92 [concrete = imports.%.be7]
// CHECK:STDOUT:   %.loc8_13.1: ref %i32 = class_element_access %.loc8_11.3, element0
// CHECK:STDOUT:   %.loc8_13.2: %i32 = bind_value %.loc8_13.1
// CHECK:STDOUT:   return %.loc8_13.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %X.75b6d8.2: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %C.elem.3f4: type = unbound_element_type %C.f2e, %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.f5f: type = struct_type {.base: %X.75b6d8.2} [symbolic]
// CHECK:STDOUT:   %complete_type.768: <witness> = complete_type_witness %struct_type.base.f5f [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %X.448: type = class_type @X, @X(%i32) [concrete]
// CHECK:STDOUT:   %G.type.862: type = fn_type @G, @X(%i32) [concrete]
// CHECK:STDOUT:   %G.d5e: %G.type.862 = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem.494: type = unbound_element_type %C.98a, %X.448 [concrete]
// CHECK:STDOUT:   %struct_type.base.d41: type = struct_type {.base: %X.448} [concrete]
// CHECK:STDOUT:   %complete_type.146: <witness> = complete_type_witness %struct_type.base.d41 [concrete]
// CHECK:STDOUT:   %G.specific_fn.7a3: <specific function> = specific_function %G.d5e, @G(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc9_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc9_9.1: type) {
// CHECK:STDOUT:   %T.loc9_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc9_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc10_19.2: type = class_type @X, @X(%T.loc9_9.2) [symbolic = %X.loc10_19.2 (constants.%X.75b6d8.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %X.loc10_19.2 [symbolic = %require_complete (constants.%require_complete.441)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc9_9.2) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %X.loc10_19.2 [symbolic = %C.elem (constants.%C.elem.3f4)]
// CHECK:STDOUT:   %struct_type.base.loc11_1.2: type = struct_type {.base: @C.%X.loc10_19.2 (%X.75b6d8.2)} [symbolic = %struct_type.base.loc11_1.2 (constants.%struct_type.base.f5f)]
// CHECK:STDOUT:   %complete_type.loc11_1.2: <witness> = complete_type_witness %struct_type.base.loc11_1.2 [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.768)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.ref: %X.type = name_ref X, file.%X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc9_9.1 [symbolic = %T.loc9_9.2 (constants.%T)]
// CHECK:STDOUT:     %X.loc10_19.1: type = class_type @X, @X(constants.%T) [symbolic = %X.loc10_19.2 (constants.%X.75b6d8.2)]
// CHECK:STDOUT:     %.loc10: @C.%C.elem (%C.elem.3f4) = base_decl %X.loc10_19.1, element0 [concrete]
// CHECK:STDOUT:     %struct_type.base.loc11_1.1: type = struct_type {.base: %X.75b6d8.2} [symbolic = %struct_type.base.loc11_1.2 (constants.%struct_type.base.f5f)]
// CHECK:STDOUT:     %complete_type.loc11_1.1: <witness> = complete_type_witness %struct_type.base.loc11_1.1 [symbolic = %complete_type.loc11_1.2 (constants.%complete_type.768)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc11_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc10
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend %X.loc10_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc16_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc16_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc16_22: %G.type.862 = specific_constant @X.%G.decl, @X(constants.%i32) [concrete = constants.%G.d5e]
// CHECK:STDOUT:   %G.ref: %G.type.862 = name_ref G, %.loc16_22 [concrete = constants.%G.d5e]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn.7a3]
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn()
// CHECK:STDOUT:   %.loc16_10: type = splice_block %i32.loc16_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc16_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc16_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc16_25.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc16_25.2: ref %i32 = temporary %.loc16_25.1, %G.call
// CHECK:STDOUT:   %i: ref %i32 = bind_name i, %.loc16_25.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc9_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc9_9.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc10_19.2 => constants.%X.448
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.494
// CHECK:STDOUT:   %struct_type.base.loc11_1.2 => constants.%struct_type.base.d41
// CHECK:STDOUT:   %complete_type.loc11_1.2 => constants.%complete_type.146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %G.type.56f312.1: type = fn_type @G, @X(%U) [symbolic]
// CHECK:STDOUT:   %G.b504c4.1: %G.type.56f312.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %G.type.862: type = fn_type @G, @X(%i32) [concrete]
// CHECK:STDOUT:   %G.d5e: %G.type.862 = struct_value () [concrete]
// CHECK:STDOUT:   %G.specific_fn.7a3: <specific function> = specific_function %G.d5e, @G(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//extend_generic_symbolic_base, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.import_ref.b8a: @X.%G.type (%G.type.56f312.1) = import_ref Main//extend_generic_symbolic_base, loc5_15, loaded [symbolic = @X.%G (constants.%G.b504c4.1)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc8_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc8_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc8_22: %G.type.862 = specific_constant imports.%Main.import_ref.b8a, @X(constants.%i32) [concrete = constants.%G.d5e]
// CHECK:STDOUT:   %G.ref: %G.type.862 = name_ref G, %.loc8_22 [concrete = constants.%G.d5e]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn.7a3]
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn()
// CHECK:STDOUT:   %.loc8_10: type = splice_block %i32.loc8_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc8_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc8_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc8_25.1: ref %i32 = temporary_storage
// CHECK:STDOUT:   %.loc8_25.2: ref %i32 = temporary %.loc8_25.1, %G.call
// CHECK:STDOUT:   %i: ref %i32 = bind_name i, %.loc8_25.2
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
