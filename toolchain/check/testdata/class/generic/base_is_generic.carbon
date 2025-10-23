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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Base.type: type = generic_class_type @Base [concrete]
// CHECK:STDOUT:   %Base.generic: %Base.type = struct_value () [concrete]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T.8b3) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T.8b3 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T.8b3} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.0f9: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %Base.7a8: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %Base.elem.d1f: type = unbound_element_type %Base.7a8, %Param [concrete]
// CHECK:STDOUT:   %struct_type.x.975: type = struct_type {.x: %Param} [concrete]
// CHECK:STDOUT:   %complete_type.db3: <witness> = complete_type_witness %struct_type.x.975 [concrete]
// CHECK:STDOUT:   %Derived.elem: type = unbound_element_type %Derived, %Base.7a8 [concrete]
// CHECK:STDOUT:   %struct_type.base.8bc: type = struct_type {.base: %Base.7a8} [concrete]
// CHECK:STDOUT:   %complete_type.b07: <witness> = complete_type_witness %struct_type.base.8bc [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %DoubleFieldAccess.type: type = fn_type @DoubleFieldAccess [concrete]
// CHECK:STDOUT:   %DoubleFieldAccess: %DoubleFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.afd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6cd: %Int.as.Copy.impl.Op.type.afd = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.a32: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.276: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f59: %Int.as.Copy.impl.Op.type.276 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.a32) [concrete]
// CHECK:STDOUT:   %.7fa: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.f59, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d0f6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.afd) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6cd)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Core.import_ref.d0f6), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_17.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %Derived.decl: type = class_decl @Derived [concrete = constants.%Derived] {} {}
// CHECK:STDOUT:   %DoubleFieldAccess.decl: %DoubleFieldAccess.type = fn_decl @DoubleFieldAccess [concrete = constants.%DoubleFieldAccess] {
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = value_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_17.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_17.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T.loc4_17.1) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T.loc4_17.1 [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T.loc4_17.1 (%T.8b3)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type.loc6_1.2: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_17.2 [symbolic = %T.loc4_17.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %.loc5: @Base.%Base.elem (%Base.elem.9af) = field_decl x, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc6_1.1: <witness> = complete_type_witness constants.%struct_type.x.2ac [symbolic = %complete_type.loc6_1.2 (constants.%complete_type.433)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc6_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Base.370
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .x = %.loc5
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
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
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(constants.%Param) [concrete = constants.%Base.7a8]
// CHECK:STDOUT:   %.loc13: %Derived.elem = base_decl %Base, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.base.8bc [concrete = constants.%complete_type.b07]
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
// CHECK:STDOUT: fn @DoubleFieldAccess(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.d1f = name_ref x, @Base.%.loc5 [concrete = @Base.%.loc5]
// CHECK:STDOUT:   %.loc17_11.1: ref %Base.7a8 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc17_11.2: ref %Base.7a8 = converted %d.ref, %.loc17_11.1
// CHECK:STDOUT:   %.loc17_11.3: ref %Param = class_element_access %.loc17_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc17_13.1: ref %i32 = class_element_access %.loc17_11.3, element0
// CHECK:STDOUT:   %.loc17_13.2: %i32 = bind_value %.loc17_13.1
// CHECK:STDOUT:   %impl.elem0: %.7fa = impl_witness_access constants.%Copy.impl_witness.a32, element0 [concrete = constants.%Int.as.Copy.impl.Op.f59]
// CHECK:STDOUT:   %bound_method.loc17_13.1: <bound method> = bound_method %.loc17_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc17_13.2: <bound method> = bound_method %.loc17_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc17_13.2(%.loc17_13.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.0f9
// CHECK:STDOUT:   %Base => constants.%Base.7a8
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.d1f
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.975
// CHECK:STDOUT:   %complete_type.loc6_1.2 => constants.%complete_type.db3
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
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %struct_type.y: type = struct_type {.y: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.09d: <witness> = complete_type_witness %struct_type.y [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %struct_type.x.2ac: type = struct_type {.x: %T.8b3} [symbolic]
// CHECK:STDOUT:   %complete_type.433: <witness> = complete_type_witness %struct_type.x.2ac [symbolic]
// CHECK:STDOUT:   %Base.370: type = class_type @Base, @Base(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Base.7a8: type = class_type @Base, @Base(%Param) [concrete]
// CHECK:STDOUT:   %struct_type.base.8bc: type = struct_type {.base: %Base.7a8} [concrete]
// CHECK:STDOUT:   %complete_type.b07: <witness> = complete_type_witness %struct_type.base.8bc [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Base.elem.9af: type = unbound_element_type %Base.370, %T.8b3 [symbolic]
// CHECK:STDOUT:   %Base.elem.d1f: type = unbound_element_type %Base.7a8, %Param [concrete]
// CHECK:STDOUT:   %struct_type.x.975: type = struct_type {.x: %Param} [concrete]
// CHECK:STDOUT:   %complete_type.db3: <witness> = complete_type_witness %struct_type.x.975 [concrete]
// CHECK:STDOUT:   %pattern_type.fb9: type = pattern_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type.501: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess.type: type = fn_type @ImportedDoubleFieldAccess [concrete]
// CHECK:STDOUT:   %ImportedDoubleFieldAccess: %ImportedDoubleFieldAccess.type = struct_value () [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ea1: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.d6b: %Int.as.Copy.impl.Op.type.ea1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.d0b: <witness> = impl_witness imports.%Copy.impl_witness_table.608, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.edd: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.99d: %Int.as.Copy.impl.Op.type.edd = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.d0b) [concrete]
// CHECK:STDOUT:   %.0ec: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.99d, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.446 = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.a92: %Param.elem = import_ref Main//extend_generic_base, loc9_8, loaded [concrete = %.be7]
// CHECK:STDOUT:   %Main.import_ref.5ab: type = import_ref Main//extend_generic_base, loc4_17, loaded [symbolic = @Base.%T (constants.%T.8b3)]
// CHECK:STDOUT:   %Main.import_ref.b5f: <witness> = import_ref Main//extend_generic_base, loc6_1, loaded [symbolic = @Base.%complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:   %Main.import_ref.8e0 = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.7f7: @Base.%Base.elem (%Base.elem.9af) = import_ref Main//extend_generic_base, loc5_8, loaded [concrete = %.e66]
// CHECK:STDOUT:   %Main.import_ref.bd0: <witness> = import_ref Main//extend_generic_base, loc14_1, loaded [concrete = constants.%complete_type.b07]
// CHECK:STDOUT:   %Main.import_ref.f6c = import_ref Main//extend_generic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.d24 = import_ref Main//extend_generic_base, loc13_27, unloaded
// CHECK:STDOUT:   %Main.import_ref.77a301.2: type = import_ref Main//extend_generic_base, loc13_26, loaded [concrete = constants.%Base.7a8]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %.e66: @Base.%Base.elem (%Base.elem.9af) = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %.be7: %Param.elem = field_decl y, element0 [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.d8e: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ea1) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.d6b)]
// CHECK:STDOUT:   %Copy.impl_witness_table.608 = impl_witness_table (%Core.import_ref.d8e), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:     %d.patt: %pattern_type.fb9 = value_binding_pattern d [concrete]
// CHECK:STDOUT:     %d.param_patt: %pattern_type.fb9 = value_param_pattern %d.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.501 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.501 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %d.param: %Derived = value_param call_param0
// CHECK:STDOUT:     %Derived.ref: type = name_ref Derived, imports.%Main.Derived [concrete = constants.%Derived]
// CHECK:STDOUT:     %d: %Derived = value_binding d, %d.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Derived [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.bd0
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.f6c
// CHECK:STDOUT:   .base = imports.%Main.import_ref.d24
// CHECK:STDOUT:   .x = <poisoned>
// CHECK:STDOUT:   extend imports.%Main.import_ref.77a301.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   complete_type_witness = imports.%Main.import_ref.e8d
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = imports.%Main.import_ref.446
// CHECK:STDOUT:   .y = imports.%Main.import_ref.a92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Base(imports.%Main.import_ref.5ab: type) [from "extend_generic_base.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Base: type = class_type @Base, @Base(%T) [symbolic = %Base (constants.%Base.370)]
// CHECK:STDOUT:   %Base.elem: type = unbound_element_type %Base, %T [symbolic = %Base.elem (constants.%Base.elem.9af)]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: @Base.%T (%T.8b3)} [symbolic = %struct_type.x (constants.%struct_type.x.2ac)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x [symbolic = %complete_type (constants.%complete_type.433)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.b5f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.8e0
// CHECK:STDOUT:     .x = imports.%Main.import_ref.7f7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @ImportedDoubleFieldAccess(%d.param: %Derived) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %d.ref: %Derived = name_ref d, %d
// CHECK:STDOUT:   %x.ref: %Base.elem.d1f = name_ref x, imports.%Main.import_ref.7f7 [concrete = imports.%.e66]
// CHECK:STDOUT:   %.loc7_11.1: ref %Base.7a8 = class_element_access %d.ref, element0
// CHECK:STDOUT:   %.loc7_11.2: ref %Base.7a8 = converted %d.ref, %.loc7_11.1
// CHECK:STDOUT:   %.loc7_11.3: ref %Param = class_element_access %.loc7_11.2, element0
// CHECK:STDOUT:   %y.ref: %Param.elem = name_ref y, imports.%Main.import_ref.a92 [concrete = imports.%.be7]
// CHECK:STDOUT:   %.loc7_13.1: ref %i32 = class_element_access %.loc7_11.3, element0
// CHECK:STDOUT:   %.loc7_13.2: %i32 = bind_value %.loc7_13.1
// CHECK:STDOUT:   %impl.elem0: %.0ec = impl_witness_access constants.%Copy.impl_witness.d0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.99d]
// CHECK:STDOUT:   %bound_method.loc7_13.1: <bound method> = bound_method %.loc7_13.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_13.2: <bound method> = bound_method %.loc7_13.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc7_13.2(%.loc7_13.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%T.8b3) {
// CHECK:STDOUT:   %T => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Base(constants.%Param) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.09d
// CHECK:STDOUT:   %Base => constants.%Base.7a8
// CHECK:STDOUT:   %Base.elem => constants.%Base.elem.d1f
// CHECK:STDOUT:   %struct_type.x => constants.%struct_type.x.975
// CHECK:STDOUT:   %complete_type => constants.%complete_type.db3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_extend_symbolic_base.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G [concrete]
// CHECK:STDOUT:   %X.G: %X.G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.fac: type = class_type @C, @C(%X) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
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
// CHECK:STDOUT:     .Self = constants.%C.f2e
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
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%X) [concrete = constants.%C.fac]
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
// CHECK:STDOUT:   %.Self: %type = bind_symbolic_name .Self [symbolic_self]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X.75b6d8.1: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %X.G.type.56f312.1: type = fn_type @X.G, @X(%U) [symbolic]
// CHECK:STDOUT:   %X.G.b504c4.1: %X.G.type.56f312.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %X.G.specific_fn.169: <specific function> = specific_function %X.G.b504c4.1, @X.G(%U) [symbolic]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %X.75b6d8.2: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.type.56f312.2: type = fn_type @X.G, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.b504c4.2: %X.G.type.56f312.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %C.elem.3f4: type = unbound_element_type %C.f2e, %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.f5f: type = struct_type {.base: %X.75b6d8.2} [symbolic]
// CHECK:STDOUT:   %complete_type.768: <witness> = complete_type_witness %struct_type.base.f5f [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %X.448: type = class_type @X, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.type.862: type = fn_type @X.G, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.d5e: %X.G.type.862 = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem.494: type = unbound_element_type %C.98a, %X.448 [concrete]
// CHECK:STDOUT:   %struct_type.base.d41: type = struct_type {.base: %X.448} [concrete]
// CHECK:STDOUT:   %complete_type.146: <witness> = complete_type_witness %struct_type.base.d41 [concrete]
// CHECK:STDOUT:   %X.G.specific_fn.7a3: <specific function> = specific_function %X.G.d5e, @X.G(%i32) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc4_14.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_14.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = bind_symbolic_name .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc8_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc4_14.2: type) {
// CHECK:STDOUT:   %U.loc4_14.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc4_14.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U.loc4_14.1) [symbolic = %X.G.type (constants.%X.G.type.56f312.1)]
// CHECK:STDOUT:   %X.G: @X.%X.G.type (%X.G.type.56f312.1) = struct_value () [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.G.decl: @X.%X.G.type (%X.G.type.56f312.1) = fn_decl @X.G [symbolic = @X.%X.G (constants.%X.G.b504c4.1)] {
// CHECK:STDOUT:       %return.patt: @X.G.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @X.G.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc4_14.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %return.param: ref @X.G.%U (%U) = out_param call_param0
// CHECK:STDOUT:       %return: ref @X.G.%U (%U) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X.75b6d8.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .G = %X.G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc8_9.2: type) {
// CHECK:STDOUT:   %T.loc8_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2: type = class_type @X, @X(%T.loc8_9.1) [symbolic = %X.loc9_19.2 (constants.%X.75b6d8.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %X.loc9_19.2 [symbolic = %require_complete (constants.%require_complete.441)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T.loc8_9.1) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %X.loc9_19.2 [symbolic = %C.elem (constants.%C.elem.3f4)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X.loc9_19.2 (%X.75b6d8.2)} [symbolic = %struct_type.base (constants.%struct_type.base.f5f)]
// CHECK:STDOUT:   %complete_type.loc10_1.2: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.768)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %X.ref: %X.type = name_ref X, file.%X.decl [concrete = constants.%X.generic]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc8_9.2 [symbolic = %T.loc8_9.1 (constants.%T)]
// CHECK:STDOUT:     %X.loc9_19.1: type = class_type @X, @X(constants.%T) [symbolic = %X.loc9_19.2 (constants.%X.75b6d8.2)]
// CHECK:STDOUT:     %.loc9: @C.%C.elem (%C.elem.3f4) = base_decl %X.loc9_19.1, element0 [concrete]
// CHECK:STDOUT:     %complete_type.loc10_1.1: <witness> = complete_type_witness constants.%struct_type.base.f5f [symbolic = %complete_type.loc10_1.2 (constants.%complete_type.768)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc10_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:     .X = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .base = %.loc9
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend %X.loc9_19.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.G(@X.%U.loc4_14.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.56f312.1)]
// CHECK:STDOUT:   %X.G: @X.G.%X.G.type (%X.G.type.56f312.1) = struct_value () [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2: <specific function> = specific_function %X.G, @X.G(%U) [symbolic = %X.G.specific_fn.loc5_24.2 (constants.%X.G.specific_fn.169)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @X.G.%U (%U) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %.loc5_24: @X.G.%X.G.type (%X.G.type.56f312.1) = specific_constant @X.%X.G.decl, @X(constants.%U) [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:     %G.ref: @X.G.%X.G.type (%X.G.type.56f312.1) = name_ref G, %.loc5_24 [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:     %X.G.specific_fn.loc5_24.1: <specific function> = specific_function %G.ref, @X.G(constants.%U) [symbolic = %X.G.specific_fn.loc5_24.2 (constants.%X.G.specific_fn.169)]
// CHECK:STDOUT:     %.loc5_10: ref @X.G.%U (%U) = splice_block %return {}
// CHECK:STDOUT:     %X.G.call: init @X.G.%U (%U) = call %X.G.specific_fn.loc5_24.1() to %.loc5_10
// CHECK:STDOUT:     return %X.G.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %i.patt: %pattern_type.7ce = value_binding_pattern i [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc13_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc13_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc13_22: %X.G.type.862 = specific_constant @X.%X.G.decl, @X(constants.%i32) [concrete = constants.%X.G.d5e]
// CHECK:STDOUT:   %G.ref: %X.G.type.862 = name_ref G, %.loc13_22 [concrete = constants.%X.G.d5e]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %G.ref, @X.G(constants.%i32) [concrete = constants.%X.G.specific_fn.7a3]
// CHECK:STDOUT:   %X.G.call: init %i32 = call %X.G.specific_fn()
// CHECK:STDOUT:   %.loc13_10: type = splice_block %i32.loc13_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc13_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc13_25.1: %i32 = value_of_initializer %X.G.call
// CHECK:STDOUT:   %.loc13_25.2: %i32 = converted %X.G.call, %.loc13_25.1
// CHECK:STDOUT:   %i: %i32 = value_binding i, %.loc13_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc4_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.1
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.1
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.1
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2 => constants.%X.G.specific_fn.169
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
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.2
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T.loc8_9.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.loc9_19.2 => constants.%X.448
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.494
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.d41
// CHECK:STDOUT:   %complete_type.loc10_1.2 => constants.%complete_type.146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U.loc4_14.1 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.862
// CHECK:STDOUT:   %X.G => constants.%X.G.d5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.862
// CHECK:STDOUT:   %X.G => constants.%X.G.d5e
// CHECK:STDOUT:   %X.G.specific_fn.loc5_24.2 => constants.%X.G.specific_fn.7a3
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %X.75b6d8.2: type = class_type @X, @X(%T) [symbolic]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %C.elem.3f4: type = unbound_element_type %C.f2e, %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %struct_type.base.f5f: type = struct_type {.base: %X.75b6d8.2} [symbolic]
// CHECK:STDOUT:   %complete_type.768: <witness> = complete_type_witness %struct_type.base.f5f [symbolic]
// CHECK:STDOUT:   %X.G.type.56f312.1: type = fn_type @X.G, @X(%U) [symbolic]
// CHECK:STDOUT:   %X.G.b504c4.1: %X.G.type.56f312.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X.75b6d8.2 [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %X.G.specific_fn.169: <specific function> = specific_function %X.G.b504c4.1, @X.G(%U) [symbolic]
// CHECK:STDOUT:   %X.G.type.56f312.2: type = fn_type @X.G, @X(%T) [symbolic]
// CHECK:STDOUT:   %X.G.b504c4.2: %X.G.type.56f312.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %C.98a: type = class_type @C, @C(%i32) [concrete]
// CHECK:STDOUT:   %X.448: type = class_type @X, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.type.862: type = fn_type @X.G, @X(%i32) [concrete]
// CHECK:STDOUT:   %X.G.d5e: %X.G.type.862 = struct_value () [concrete]
// CHECK:STDOUT:   %C.elem.494: type = unbound_element_type %C.98a, %X.448 [concrete]
// CHECK:STDOUT:   %struct_type.base.d41: type = struct_type {.base: %X.448} [concrete]
// CHECK:STDOUT:   %complete_type.146: <witness> = complete_type_witness %struct_type.base.d41 [concrete]
// CHECK:STDOUT:   %X.G.specific_fn.7a3: <specific function> = specific_function %X.G.d5e, @X.G(%i32) [concrete]
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
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.1: type = import_ref Main//extend_generic_symbolic_base, loc4_14, loaded [symbolic = @X.%U (constants.%U)]
// CHECK:STDOUT:   %Main.import_ref.8f2: <witness> = import_ref Main//extend_generic_symbolic_base, loc6_1, loaded [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   %Main.import_ref.e8e = import_ref Main//extend_generic_symbolic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.b8a: @X.%X.G.type (%X.G.type.56f312.1) = import_ref Main//extend_generic_symbolic_base, loc5_15, loaded [symbolic = @X.%X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.2: type = import_ref Main//extend_generic_symbolic_base, loc8_9, loaded [symbolic = @C.%T (constants.%T)]
// CHECK:STDOUT:   %Main.import_ref.93f: <witness> = import_ref Main//extend_generic_symbolic_base, loc10_1, loaded [symbolic = @C.%complete_type (constants.%complete_type.768)]
// CHECK:STDOUT:   %Main.import_ref.4c0 = import_ref Main//extend_generic_symbolic_base, inst{{[0-9A-F]+}} [no loc], unloaded
// CHECK:STDOUT:   %Main.import_ref.65d = import_ref Main//extend_generic_symbolic_base, loc9_20, unloaded
// CHECK:STDOUT:   %Main.import_ref.561eb2.2: type = import_ref Main//extend_generic_symbolic_base, loc9_19, loaded [symbolic = @C.%X (constants.%X.75b6d8.2)]
// CHECK:STDOUT:   %Main.import_ref.5ab3ec.3: type = import_ref Main//extend_generic_symbolic_base, loc4_14, loaded [symbolic = @X.%U (constants.%U)]
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
// CHECK:STDOUT: generic class @C(imports.%Main.import_ref.5ab3ec.2: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X: type = class_type @X, @X(%T) [symbolic = %X (constants.%X.75b6d8.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %X [symbolic = %require_complete (constants.%require_complete.441)]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic = %C (constants.%C.f2e)]
// CHECK:STDOUT:   %C.elem: type = unbound_element_type %C, %X [symbolic = %C.elem (constants.%C.elem.3f4)]
// CHECK:STDOUT:   %struct_type.base: type = struct_type {.base: @C.%X (%X.75b6d8.2)} [symbolic = %struct_type.base (constants.%struct_type.base.f5f)]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.base [symbolic = %complete_type (constants.%complete_type.768)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.93f
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.4c0
// CHECK:STDOUT:     .base = imports.%Main.import_ref.65d
// CHECK:STDOUT:     .G = <poisoned>
// CHECK:STDOUT:     extend imports.%Main.import_ref.561eb2.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(imports.%Main.import_ref.5ab3ec.1: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.56f312.1)]
// CHECK:STDOUT:   %X.G: @X.%X.G.type (%X.G.type.56f312.1) = struct_value () [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     complete_type_witness = imports.%Main.import_ref.8f2
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Main.import_ref.e8e
// CHECK:STDOUT:     .G = imports.%Main.import_ref.b8a
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %j.patt: %pattern_type.7ce = value_binding_pattern j [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, imports.%Main.C [concrete = constants.%C.generic]
// CHECK:STDOUT:   %int_32.loc7_18: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc7_18: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %C: type = class_type @C, @C(constants.%i32) [concrete = constants.%C.98a]
// CHECK:STDOUT:   %.loc7_22: %X.G.type.862 = specific_constant imports.%Main.import_ref.b8a, @X(constants.%i32) [concrete = constants.%X.G.d5e]
// CHECK:STDOUT:   %G.ref: %X.G.type.862 = name_ref G, %.loc7_22 [concrete = constants.%X.G.d5e]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %G.ref, @X.G(constants.%i32) [concrete = constants.%X.G.specific_fn.7a3]
// CHECK:STDOUT:   %X.G.call: init %i32 = call %X.G.specific_fn()
// CHECK:STDOUT:   %.loc7_10: type = splice_block %i32.loc7_10 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc7_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc7_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7_25.1: %i32 = value_of_initializer %X.G.call
// CHECK:STDOUT:   %.loc7_25.2: %i32 = converted %X.G.call, %.loc7_25.1
// CHECK:STDOUT:   %j: %i32 = value_binding j, %.loc7_25.2
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.G(imports.%Main.import_ref.5ab3ec.3: type) [from "extend_generic_symbolic_base.carbon"] {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %U [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %U [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %X.G.type: type = fn_type @X.G, @X(%U) [symbolic = %X.G.type (constants.%X.G.type.56f312.1)]
// CHECK:STDOUT:   %X.G: @X.G.%X.G.type (%X.G.type.56f312.1) = struct_value () [symbolic = %X.G (constants.%X.G.b504c4.1)]
// CHECK:STDOUT:   %X.G.specific_fn: <specific function> = specific_function %X.G, @X.G(%U) [symbolic = %X.G.specific_fn (constants.%X.G.specific_fn.169)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.1
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%T) {
// CHECK:STDOUT:   %U => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.2
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.56f312.1
// CHECK:STDOUT:   %X.G => constants.%X.G.b504c4.1
// CHECK:STDOUT:   %X.G.specific_fn => constants.%X.G.specific_fn.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%i32) {
// CHECK:STDOUT:   %T => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X => constants.%X.448
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %C => constants.%C.98a
// CHECK:STDOUT:   %C.elem => constants.%C.elem.494
// CHECK:STDOUT:   %struct_type.base => constants.%struct_type.base.d41
// CHECK:STDOUT:   %complete_type => constants.%complete_type.146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.862
// CHECK:STDOUT:   %X.G => constants.%X.G.d5e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.G(constants.%i32) {
// CHECK:STDOUT:   %U => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %X.G.type => constants.%X.G.type.862
// CHECK:STDOUT:   %X.G => constants.%X.G.d5e
// CHECK:STDOUT:   %X.G.specific_fn => constants.%X.G.specific_fn.7a3
// CHECK:STDOUT: }
// CHECK:STDOUT:
