// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
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

class C(T:! type) {
  extend base: X(T);
}

fn F() {
  let unused i: i32 = C(i32).G();
}


// --- import_extend_generic_symbolic_base.carbon

library "[[@TEST_NAME]]";

import library "extend_generic_symbolic_base";

fn H() {
  let unused j: i32 = C(i32).G();
}

// CHECK:STDOUT: --- extend_generic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T.67d) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Base.elem.8ab: type = unbound_element_type %Base.d0c, %T.67d [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0f9: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.f8f: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %Base.elem.6d8: type = unbound_element_type %Base.f8f, %Param [concrete]
// CHECK:STDOUT:   %struct_type.x.d96: type = struct_type {.x: %Param} [concrete]
// CHECK:STDOUT:   %complete_type.95c: <witness> = complete_type_witness %struct_type.x.d96 [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.f8f [concrete]
// CHECK:STDOUT:   %struct_type.base.9a9: type = struct_type {.base: %Base.f8f} [concrete]
// CHECK:STDOUT:   %complete_type.8de: <witness> = complete_type_witness %struct_type.base.9a9 [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %DoubleFieldAccess.type: type = fn_type @DoubleFieldAccess [concrete]
// CHECK:STDOUT:   %DoubleFieldAccess: %DoubleFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Base = %Base.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .Derived = %Derived.decl
// CHECK:STDOUT:     .DoubleFieldAccess = %DoubleFieldAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Base.decl: %Base.type = class_decl @Base [concrete = constants.%Base.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_21.1: type = splice_block %.loc4_21.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_21.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %DoubleFieldAccess.decl: %DoubleFieldAccess.type = fn_decl @DoubleFieldAccess [concrete = constants.%DoubleFieldAccess] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.9f6 = at_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc16: Core.Form = init_form %i32 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_17.1 [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.1) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.1 [symbolic = %Base.elem (constants.%Base.elem.8ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T.loc4_17.1 (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.2 [symbolic = %T.loc4_17.1 (constants.%T.67d)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.8ab) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.0c5 [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.735)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.d0c
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.y [concrete = constants.%complete_type.0f9]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .y = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived {
// CHECK:STDOUT:   %Base.ref: %Base.type = name_ref Base, file.%Base.decl [concrete = constants.%Base.generic]
// CHECK:STDOUT:   %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [concrete = constants.%Base.f8f]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.9a9 [concrete = constants.%complete_type.8de]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Derived
// CHECK:STDOUT:   .Base = <poisoned>
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .base = %.loc13
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend %Base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @DoubleFieldAccess(%d.param: %Derived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.6d8 = name_ref x, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc17_11.1: ref %Base.f8f = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc17_11.2: ref %Base.f8f = converted %d.ref, %.loc17_11.1
// CHECK:STDOUT:   %.loc17_11.3: ref %Param = class_element_access %.loc17_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc17_13.1: ref %i32 = class_element_access %.loc17_11.3, element0
// CHECK:STDOUT:   %.loc17_13.2: %i32 = acquire_value %.loc17_13.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc17_13.1: <bound method> = bound_method %.loc17_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_13.2: <bound method> = bound_method %.loc17_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc17_13.2(%.loc17_13.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.0f9
// CHECK:STDOUT:   %Base => constants.%Base.f8f
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.6d8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.d96
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.95c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.09d: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.0c5: type = struct_type {.x: %T.67d} [symbolic]
// CHECK:STDOUT:   %complete_type.735: <witness> = complete_type_witness %struct_type.x.0c5 [symbolic]
// CHECK:STDOUT:   %Base.d0c: type = class_type @Base, @Base(%T.67d) [symbolic]
// CHECK:STDOUT:   %Base.elem.8ab: type = unbound_element_type %Base.d0c, %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %Base.f8f: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %struct_type.x.d96: type = struct_type {.x: %Param} [concrete]
// CHECK:STDOUT:   %complete_type.95c: <witness> = complete_type_witness %struct_type.x.d96 [concrete]
// CHECK:STDOUT:   %Base.elem.6d8: type = unbound_element_type %Base.f8f, %Param [concrete]
// CHECK:STDOUT:   %struct_type.base.9a9: type = struct_type {.base: %Base.f8f} [concrete]
// CHECK:STDOUT:   %complete_type.8de: <witness> = complete_type_witness %struct_type.base.9a9 [concrete]
// CHECK:STDOUT:   %pattern_type.9f6: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %.4ca: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %pattern_type.501: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.type: type = fn_type @ImportedDoubleFieldAccess [concrete]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess: %ImportedDoubleFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.08e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.014: %Int.as.Copy.impl.Op.type.08e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a2f: <witness> = impl_witness imports.%Copy.impl_witness_table.d6d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.837: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.2b1: %Int.as.Copy.impl.Op.type.837 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a2f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.d31: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.2b1, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.Base = import_ref Main//extend_generic_base, Base, unloaded
// CHECK:STDOUT:   %Main.Param = import_ref Main//extend_generic_base, Param, unloaded
// CHECK:STDOUT:   %Main.Derived: type = import_ref Main//extend_generic_base, Derived, loaded [concrete = constants.%Derived]
// CHECK:STDOUT:   %Main.DoubleFieldAccess = import_ref Main//extend_generic_base, DoubleFieldAccess, unloaded
// CHECK:STDOUT:   %Core.ece: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Main.import_ref.e8d: <witness> = import_ref Main//extend_generic_base, loc10_1, loaded [concrete = constants.%complete_type.09d]
// CHECK:STDOUT:   %Main.import_ref.304 = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.0c2: %Param.elem = import_ref Main//extend_generic_base, loc9_8, loaded [concrete = %.e89]
// CHECK:STDOUT:   %Main.import_ref.7e3: <witness> = import_ref Main//extend_generic_base, loc6_1, loaded [symbolic = @Base.%complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:   %Main.import_ref.2af = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.568: @Base.%Base.elem (%Base.elem.8ab) = import_ref Main//extend_generic_base, loc5_8, loaded [concrete = %.bef]
// CHECK:STDOUT:   %Main.import_ref.b3b: type = import_ref Main//extend_generic_base, loc4_17, loaded [symbolic = @Base.%T (constants.%T.67d)]
// CHECK:STDOUT:   %Main.import_ref.56c: <witness> = import_ref Main//extend_generic_base, loc14_1, loaded [concrete = constants.%complete_type.8de]
// CHECK:STDOUT:   %Main.import_ref.33c = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.4b4 = import_ref Main//extend_generic_base, loc13_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.8875f0.2: type = import_ref Main//extend_generic_base, loc13_26, loaded [concrete = constants.%Base.f8f]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.bef: @Base.%Base.elem (%Base.elem.8ab) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %.e89: %Param.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.ea6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.08e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.014)]
// CHECK:STDOUT:   %Copy.impl_witness_table.d6d = impl_witness_table (%Core.import_ref.ea6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Base = imports.%Main.Base
// CHECK:STDOUT:     .Param = imports.%Main.Param
// CHECK:STDOUT:     .Derived = imports.%Main.Derived
// CHECK:STDOUT:     .DoubleFieldAccess = imports.%Main.DoubleFieldAccess
// CHECK:STDOUT:     .Core = imports.%Core.ece
// CHECK:STDOUT:     .ImportedDoubleFieldAccess = %ImportedDoubleFieldAccess.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.decl: %ImportedDoubleFieldAccess.type = fn_decl @ImportedDoubleFieldAccess [concrete = constants.%ImportedDoubleFieldAccess] {
// CHECK:STDOUT:     %d.param_patt: %pattern_type.9f6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %d.patt: %pattern_type.9f6 = at_binding_pattern d, %d.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.501 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.501 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6: Core.Form = init_form %i32 [concrete = constants.%.4ca]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, imports.%Main.Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.56c
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.33c
// CHECK:STDOUT:   .base = imports.%Main.import_ref.4b4
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.8875f0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.e8d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.304
// CHECK:STDOUT:   .y = imports.%Main.import_ref.0c2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(imports.%Main.import_ref.b3b: type) [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.d0c)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T [symbolic = %Base.elem (constants.%Base.elem.8ab)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T (%T.67d)} [symbolic = %struct_type.x (constants.%struct_type.x.0c5)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.735)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.7e3
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.2af
// CHECK:STDOUT:     .x = imports.%Main.import_ref.568
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedDoubleFieldAccess(%d.param: %Derived) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.6d8 = name_ref x, imports.%Main.import_ref.568 [concrete = imports.%.bef]
// CHECK:STDOUT:   %.loc7_11.1: ref %Base.f8f = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc7_11.2: ref %Base.f8f = converted %d.ref, %.loc7_11.1
// CHECK:STDOUT:   %.loc7_11.3: ref %Param = class_element_access %.loc7_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, imports.%Main.import_ref.0c2 [concrete = imports.%.e89]
// CHECK:STDOUT:   %.loc7_13.1: ref %i32 = class_element_access %.loc7_11.3, element0
// CHECK:STDOUT:   %.loc7_13.2: %i32 = acquire_value %.loc7_13.1
// CHECK:STDOUT:   %impl.elem0: %.d31 = impl_witness_access constants.%Copy.impl_witness.a2f, element0 [concrete = constants.%Int.as.Copy.impl.Op.2b1]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %.loc7_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %.loc7_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_13.2(%.loc7_13.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T.67d) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.09d
// CHECK:STDOUT:   %Base => constants.%Base.f8f
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.6d8
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.d96
// CHECK:STDOUT:   %complete_type => constants.%complete_type.95c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G [concrete]
// CHECK:STDOUT:   %X.G: %X.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.513: type = class_type @C, @C(%X) [concrete]
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
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_9.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_9.2 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:     %.loc9: <error> = base_decl <error>, element0 [concrete]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness <error> [concrete = <error>]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     has_error
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @X {
// CHECK:STDOUT:   %X.G.decl: %X.G.type = fn_decl @X.G [concrete = constants.%X.G] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%X
// CHECK:STDOUT:   .G = %X.G.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @X.G() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %X.ref: type = name_ref X, file.%X.decl [concrete = constants.%X]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.513]
// CHECK:STDOUT:   %G.ref: <error> = name_ref G, <error> [concrete = <error>]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%X) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%X
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.03b463.1: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %U [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %X.G.type.20eb90.1: type = fn_type @X.G, @X(%U) [symbolic]
// CHECK:STDOUT:   %X.G.f9f685.1: %X.G.type.20eb90.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %X.G.specific_fn.974: <specific function> = specific_function %X.G.f9f685.1, @X.G(%U) [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %X.03b463.2: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.type.20eb90.2: type = fn_type @X.G, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.f9f685.2: %X.G.type.20eb90.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.5dd: <witness> = require_complete_type %X.03b463.2 [symbolic]
// CHECK:STDOUT:   %C.elem.161: type = unbound_element_type %C.5a3, %X.03b463.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.a6f: type = struct_type {.base: %X.03b463.2} [symbolic]
// CHECK:STDOUT:   %complete_type.bec: <witness> = complete_type_witness %struct_type.base.a6f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.b13: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %X.1bc: type = class_type @X, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.type.f75: type = fn_type @X.G, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.e2a: %X.G.type.f75 = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem.7f5: type = unbound_element_type %C.b13, %X.1bc [concrete]
// CHECK:STDOUT:   %struct_type.base.b52: type = struct_type {.base: %X.1bc} [concrete]
// CHECK:STDOUT:   %complete_type.ab4: <witness> = complete_type_witness %struct_type.base.b52 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %X.G.specific_fn.54d: <specific function> = specific_function %X.G.e2a, @X.G(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_18.1: type = splice_block %.loc4_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc4_14.2: type = symbolic_binding U, 0 [symbolic = %U.loc4_14.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc4_14.2: type) {
// CHECK:STDOUT:   %U.loc4_14.1: type = symbolic_binding U, 0 [symbolic = %U.loc4_14.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U.loc4_14.1) [symbolic = %X.G.type (constants.%X.G.type.20eb90.1)]
// CHECK:STDOUT:   %X.G: @X.%X.G.type (%X.G.type.20eb90.1) = struct_value () [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.G.decl: @X.%X.G.type (%X.G.type.20eb90.1) = fn_decl @X.G [symbolic = @X.%X.G (constants.%X.G.f9f685.1)] {
// CHECK:STDOUT:       %return.param_patt: @X.G.%pattern_type (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @X.G.%pattern_type (%pattern_type.51d) = return_slot_pattern %return.param_patt, %U.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc4_14.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %.loc5_13.3: Core.Form = init_form %U.ref [symbolic = %.loc5_13.2 (constants.%.184)]
// CHECK:STDOUT:       %return.param: ref @X.G.%U (%U) = out_param call_param0
// CHECK:STDOUT:       %return: ref @X.G.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X.03b463.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .G = %X.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2: type = class_type @X, @X(%T.loc8_9.1) [symbolic = %X.loc9_19.2 (constants.%X.03b463.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %X.loc9_19.2 [symbolic = %require_complete (constants.%require_complete.5dd)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc8_9.1) [symbolic = %C (constants.%C.5a3)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %X.loc9_19.2 [symbolic = %C.elem (constants.%C.elem.161)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X.loc9_19.2 (%X.03b463.2)} [symbolic = %struct_type.base (constants.%struct_type.base.a6f)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.ref: %X.type = name_ref X, file.%X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %X.loc9_19.1: type = class_type @X, @X(constants.%T) [symbolic = %X.loc9_19.2 (constants.%X.03b463.2)]
// CHECK:STDOUT:     %.loc9: @C.%C.elem (%C.elem.161) = base_decl %X.loc9_19.1, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.a6f [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.bec)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.5a3
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend %X.loc9_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.G(@X.%U.loc4_14.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %.loc5_13.2: Core.Form = init_form %U [symbolic = %.loc5_13.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.20eb90.1)]
// CHECK:STDOUT:   %X.G: @X.G.%X.G.type (%X.G.type.20eb90.1) = struct_value () [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2: <specific function> = specific_function %X.G, @X.G(%U) [symbolic = %X.G.specific_fn.loc5_24.2 (constants.%X.G.specific_fn.974)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @X.G.%U (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_24: @X.G.%X.G.type (%X.G.type.20eb90.1) = specific_constant @X.%X.G.decl, @X(constants.%U) [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:     %G.ref: @X.G.%X.G.type (%X.G.type.20eb90.1) = name_ref G, %.loc5_24 [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:     %X.G.specific_fn.loc5_24.1: <specific function> = specific_function %G.ref, @X.G(constants.%U) [symbolic = %X.G.specific_fn.loc5_24.2 (constants.%X.G.specific_fn.974)]
// CHECK:STDOUT:     %.loc5_13.1: ref @X.G.%U (%U) = splice_block %return.param {}
// CHECK:STDOUT:     %X.G.call: init @X.G.%U (%U) to %.loc5_13.1 = call %X.G.specific_fn.loc5_24.1()
// CHECK:STDOUT:     return %X.G.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = value_binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32.loc13_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.b13]
// CHECK:STDOUT:   %.loc13_29: %X.G.type.f75 = specific_constant @X.%X.G.decl, @X(constants.%i32) [concrete = constants.%X.G.e2a]
// CHECK:STDOUT:   %G.ref: %X.G.type.f75 = name_ref G, %.loc13_29 [concrete = constants.%X.G.e2a]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %G.ref, @X.G(constants.%i32) [concrete = constants.%X.G.specific_fn.54d]
// CHECK:STDOUT:   %X.G.call: init %i32 = call %X.G.specific_fn()
// CHECK:STDOUT:   %i32.loc13_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc13_32.1: %i32 = value_of_initializer %X.G.call
// CHECK:STDOUT:   %.loc13_32.2: %i32 = converted %X.G.call, %.loc13_32.1
// CHECK:STDOUT:   %i: %i32 = value_binding i, %.loc13_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc4_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.1
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %.loc5_13.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.1
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.1
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2 => constants.%X.G.specific_fn.974
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.2
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2 => constants.%X.1bc
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.b13
// CHECK:STDOUT:   %C.elem => constants.%C.elem.7f5
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.b52
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.ab4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.f75
// CHECK:STDOUT:   %X.G => constants.%X.G.e2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %.loc5_13.2 => constants.%.ff5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.f75
// CHECK:STDOUT:   %X.G => constants.%X.G.e2a
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2 => constants.%X.G.specific_fn.54d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- import_extend_generic_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %X.G.type.20eb90.1: type = fn_type @X.G, @X(%U) [symbolic]
// CHECK:STDOUT:   %X.G.f9f685.1: %X.G.type.20eb90.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %U [symbolic]
// CHECK:STDOUT:   %X.G.specific_fn.974: <specific function> = specific_function %X.G.f9f685.1, @X.G(%U) [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %X.03b463.2: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.type.20eb90.2: type = fn_type @X.G, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.f9f685.2: %X.G.type.20eb90.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.5a3: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.161: type = unbound_element_type %C.5a3, %X.03b463.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.a6f: type = struct_type {.base: %X.03b463.2} [symbolic]
// CHECK:STDOUT:   %complete_type.bec: <witness> = complete_type_witness %struct_type.base.a6f [symbolic]
// CHECK:STDOUT:   %require_complete.5dd: <witness> = require_complete_type %X.03b463.2 [symbolic]
// CHECK:STDOUT:   %C.b13: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %X.1bc: type = class_type @X, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.type.f75: type = fn_type @X.G, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.e2a: %X.G.type.f75 = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem.7f5: type = unbound_element_type %C.b13, %X.1bc [concrete]
// CHECK:STDOUT:   %struct_type.base.b52: type = struct_type {.base: %X.1bc} [concrete]
// CHECK:STDOUT:   %complete_type.ab4: <witness> = complete_type_witness %struct_type.base.b52 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %X.G.specific_fn.54d: <specific function> = specific_function %X.G.e2a, @X.G(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.X = import_ref Main//extend_generic_symbolic_base, X, unloaded
// CHECK:STDOUT:   %Main.C: %C.type = import_ref Main//extend_generic_symbolic_base, C, loaded [concrete = constants.%C.generic]
// CHECK:STDOUT:   %Main.F = import_ref Main//extend_generic_symbolic_base, F, unloaded
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.1: type = import_ref Main//extend_generic_symbolic_base, loc4_14, loaded [symbolic = @X.%U (constants.%U)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//extend_generic_symbolic_base, loc6_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.5f5 = import_ref Main//extend_generic_symbolic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.940: @X.%X.G.type (%X.G.type.20eb90.1) = import_ref Main//extend_generic_symbolic_base, loc5_15, loaded [symbolic = @X.%X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.2: type = import_ref Main//extend_generic_symbolic_base, loc4_14, loaded [symbolic = @X.%U (constants.%U)]
// CHECK:STDOUT:   %Main.import_ref.d67: <witness> = import_ref Main//extend_generic_symbolic_base, loc10_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.bec)]
// CHECK:STDOUT:   %Main.import_ref.820 = import_ref Main//extend_generic_symbolic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.5a3 = import_ref Main//extend_generic_symbolic_base, loc9_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.7f62b4.2: type = import_ref Main//extend_generic_symbolic_base, loc9_19, loaded [symbolic = @C.%X (constants.%X.03b463.2)]
// CHECK:STDOUT:   %Main.import_ref.b3bc94.3: type = import_ref Main//extend_generic_symbolic_base, loc8_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .X = imports.%Main.X
// CHECK:STDOUT:     .C = imports.%Main.C
// CHECK:STDOUT:     .F = imports.%Main.F
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %default.import = import <none>
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.b3bc94.3: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X: type = class_type @X, @X(%T) [symbolic = %X (constants.%X.03b463.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %X [symbolic = %require_complete (constants.%require_complete.5dd)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.5a3)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %X [symbolic = %C.elem (constants.%C.elem.161)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X (%X.03b463.2)} [symbolic = %struct_type.base (constants.%struct_type.base.a6f)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type (constants.%complete_type.bec)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.d67
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.820
// CHECK:STDOUT:     .base = imports.%Main.import_ref.5a3
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend imports.%Main.import_ref.7f62b4.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(imports.%Main.import_ref.b3bc94.2: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.20eb90.1)]
// CHECK:STDOUT:   %X.G: @X.%X.G.type (%X.G.type.20eb90.1) = struct_value () [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.5f5
// CHECK:STDOUT:     .G = imports.%Main.import_ref.940
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.7ce = value_binding_pattern j [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %i32.loc7_25: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.b13]
// CHECK:STDOUT:   %.loc7_29: %X.G.type.f75 = specific_constant imports.%Main.import_ref.940, @X(constants.%i32) [concrete = constants.%X.G.e2a]
// CHECK:STDOUT:   %G.ref: %X.G.type.f75 = name_ref G, %.loc7_29 [concrete = constants.%X.G.e2a]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %G.ref, @X.G(constants.%i32) [concrete = constants.%X.G.specific_fn.54d]
// CHECK:STDOUT:   %X.G.call: init %i32 = call %X.G.specific_fn()
// CHECK:STDOUT:   %i32.loc7_17: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc7_32.1: %i32 = value_of_initializer %X.G.call
// CHECK:STDOUT:   %.loc7_32.2: %i32 = converted %X.G.call, %.loc7_32.1
// CHECK:STDOUT:   %j: %i32 = value_binding j, %.loc7_32.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.G(imports.%Main.import_ref.b3bc94.1: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %U [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U [symbolic = %pattern_type (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic = %require_complete (constants.%require_complete.944)]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.20eb90.1)]
// CHECK:STDOUT:   %X.G: @X.G.%X.G.type (%X.G.type.20eb90.1) = struct_value () [symbolic = %X.G (constants.%X.G.f9f685.1)]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %X.G, @X.G(%U) [symbolic = %X.G.specific_fn (constants.%X.G.specific_fn.974)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.1
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.944
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.1
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.1
// CHECK:STDOUT:   %X.G.specific_fn => constants.%X.G.specific_fn.974
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.20eb90.2
// CHECK:STDOUT:   %X.G => constants.%X.G.f9f685.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X => constants.%X.1bc
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.b13
// CHECK:STDOUT:   %C.elem => constants.%C.elem.7f5
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.b52
// CHECK:STDOUT:   %complete_type => constants.%complete_type.ab4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.f75
// CHECK:STDOUT:   %X.G => constants.%X.G.e2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %.1 => constants.%.ff5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.f75
// CHECK:STDOUT:   %X.G => constants.%X.G.e2a
// CHECK:STDOUT:   %X.G.specific_fn => constants.%X.G.specific_fn.54d
// CHECK:STDOUT: }
// CHECK:STDOUT:
