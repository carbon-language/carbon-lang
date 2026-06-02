// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/dependent_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/dependent_param.carbon

// --- nested_class.carbon

library "[[@TEST_NAME]]";

class Outer(T:! Core.Copy) {
  class Inner(U:! T) {
    //@dump-sem-ir-begin
    fn Get() -> T { return U; }
    //@dump-sem-ir-end
  }
}

//@dump-sem-ir-begin
var n: i32 = Outer(i32).Inner(42).Get();
//@dump-sem-ir-end

// CHECK:STDOUT: --- nested_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.f49: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.467: type = class_type @Outer, @Outer(%T.f49) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f49 [symbolic]
// CHECK:STDOUT:   %pattern_type.c0fdca.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %U.51d: %T.as_type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1f5: type = generic_class_type @Inner, @Outer(%T.f49) [symbolic]
// CHECK:STDOUT:   %Inner.generic.ac8: %Inner.type.1f5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.a12: type = class_type @Inner, @Inner(%T.f49, %U.51d) [symbolic]
// CHECK:STDOUT:   %.dedae6.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.Get.type.5c5: type = fn_type @Inner.Get, @Inner(%T.f49, %U.51d) [symbolic]
// CHECK:STDOUT:   %Inner.Get.48d: %Inner.Get.type.5c5 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.77e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2dd: <witness> = lookup_impl_witness %T.f49, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.e9f10f.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f49) [symbolic]
// CHECK:STDOUT:   %.e16: type = fn_type_with_self_type %Copy.WithSelf.Op.type.e9f10f.2, %T.f49 [symbolic]
// CHECK:STDOUT:   %impl.elem0.41e: %.e16 = impl_witness_access %Copy.lookup_impl_witness.2dd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.7ec: <specific function> = specific_impl_function %impl.elem0.41e, @Copy.WithSelf.Op(%T.f49) [symbolic]
// CHECK:STDOUT:   %bound_method.f9a: <bound method> = bound_method %U.51d, %impl.elem0.41e [symbolic]
// CHECK:STDOUT:   %bound_method.acb: <bound method> = bound_method %U.51d, %specific_impl_fn.7ec [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.14d: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Outer.0c2: type = class_type @Outer, @Outer(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Inner.type.535: type = generic_class_type @Inner, @Outer(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %Inner.generic.f35: %Inner.type.535 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.061: <witness> = impl_witness imports.%Copy.impl_witness_table.fb5 [concrete]
// CHECK:STDOUT:   %Copy.facet.23a: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.061) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.98e: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.daa: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Inner.ebc: type = class_type @Inner, @Inner(%Copy.facet.14d, %int_42.daa) [concrete]
// CHECK:STDOUT:   %Inner.Get.type.fcb: type = fn_type @Inner.Get, @Inner(%Copy.facet.14d, %int_42.daa) [concrete]
// CHECK:STDOUT:   %Inner.Get.cd1: %Inner.Get.type.fcb = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Inner.Get.cd1, @Inner.Get(%Copy.facet.14d, %int_42.daa) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.14d) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet.14d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_42.daa, %Int.as.Copy.impl.Op.ce7 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.ece: <bound method> = bound_method %int_42.daa, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.83f = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.fb5 = impl_witness_table (%Core.import_ref.83f), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.104 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.104 = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt [concrete]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.467
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type, %U.loc5_16.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type: type = fn_type @Inner.Get, @Inner(%T, %U.loc5_16.1) [symbolic = %Inner.Get.type (constants.%Inner.Get.type.5c5)]
// CHECK:STDOUT:   %Inner.Get: @Inner.%Inner.Get.type (%Inner.Get.type.5c5) = struct_value () [symbolic = %Inner.Get (constants.%Inner.Get.48d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.Get.decl: @Inner.%Inner.Get.type (%Inner.Get.type.5c5) = fn_decl @Inner.Get [symbolic = @Inner.%Inner.Get (constants.%Inner.Get.48d)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.Get.%pattern_type (%pattern_type.c0fdca.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.Get.%pattern_type (%pattern_type.c0fdca.2) = return_slot_pattern %return.param_patt, %.loc7_17.3 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:       %T.as_type.loc7_17.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_17.3: type = converted %T.ref, %T.as_type.loc7_17.2 [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_17.4: Core.Form = init_form %.loc7_17.3 [symbolic = %.loc7_17.2 (constants.%.dedae6.2)]
// CHECK:STDOUT:       %return.param: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.a12
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Inner.Get.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.Get(@Outer.%T.loc4_14.2: %Copy.type, @Inner.%U.loc5_16.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f49)]
// CHECK:STDOUT:   %T.as_type.loc7_17.1: type = facet_access_type %T [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc7_17.2: Core.Form = init_form %T.as_type.loc7_17.1 [symbolic = %.loc7_17.2 (constants.%.dedae6.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_17.1 [symbolic = %pattern_type (constants.%pattern_type.c0fdca.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_17.1 [symbolic = %require_complete (constants.%require_complete.77e)]
// CHECK:STDOUT:   %U: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = symbolic_binding U, 1 [symbolic = %U (constants.%U.51d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.e9f10f.2)]
// CHECK:STDOUT:   %.loc7_28: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc7_28 (constants.%.e16)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2dd)]
// CHECK:STDOUT:   %impl.elem0.loc7_28.2: @Inner.Get.%.loc7_28 (%.e16) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:   %bound_method.loc7_28.3: <bound method> = bound_method %U, %impl.elem0.loc7_28.2 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.f9a)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2: <specific function> = specific_impl_function %impl.elem0.loc7_28.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:   %bound_method.loc7_28.4: <bound method> = bound_method %U, %specific_impl_fn.loc7_28.2 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.acb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = name_ref U, @Inner.%U.loc5_16.2 [symbolic = %U (constants.%U.51d)]
// CHECK:STDOUT:     %impl.elem0.loc7_28.1: @Inner.Get.%.loc7_28 (%.e16) = impl_witness_access constants.%Copy.lookup_impl_witness.2dd, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.41e)]
// CHECK:STDOUT:     %bound_method.loc7_28.1: <bound method> = bound_method %U.ref, %impl.elem0.loc7_28.1 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.f9a)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_28.1: <specific function> = specific_impl_function %impl.elem0.loc7_28.1, @Copy.WithSelf.Op(constants.%T.f49) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.7ec)]
// CHECK:STDOUT:     %bound_method.loc7_28.2: <bound method> = bound_method %U.ref, %specific_impl_fn.loc7_28.1 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.acb)]
// CHECK:STDOUT:     %.loc7_17.1: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) to %.loc7_17.1 = call %bound_method.loc7_28.2(%U.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_23: %Copy.type = facet_value %i32, (constants.%Copy.impl_witness.1a4) [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:   %.loc13_23: %Copy.type = converted %i32, %Copy.facet.loc13_23 [concrete = constants.%Copy.facet.14d]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Outer.0c2]
// CHECK:STDOUT:   %.loc13_24: %Inner.type.535 = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.14d) [concrete = constants.%Inner.generic.f35]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.535 = name_ref Inner, %.loc13_24 [concrete = constants.%Inner.generic.f35]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %Copy.facet.loc13_33: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.061) [concrete = constants.%Copy.facet.23a]
// CHECK:STDOUT:   %.loc13_33.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_33 [concrete = constants.%Copy.facet.23a]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc13_33.1: <bound method> = bound_method constants.%int_42.20e, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_33.2: <bound method> = bound_method constants.%int_42.20e, %specific_fn [concrete = constants.%bound_method.98e]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_33.2(constants.%int_42.20e) [concrete = constants.%int_42.daa]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.daa]
// CHECK:STDOUT:   %.loc13_33.3: %i32 = converted constants.%int_42.20e, %.loc13_33.2 [concrete = constants.%int_42.daa]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%Copy.facet.14d, constants.%int_42.daa) [concrete = constants.%Inner.ebc]
// CHECK:STDOUT:   %.loc13_34: %Inner.Get.type.fcb = specific_constant @Inner.%Inner.Get.decl, @Inner(constants.%Copy.facet.14d, constants.%int_42.daa) [concrete = constants.%Inner.Get.cd1]
// CHECK:STDOUT:   %Get.ref: %Inner.Get.type.fcb = name_ref Get, %.loc13_34 [concrete = constants.%Inner.Get.cd1]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Get.ref, @Inner.Get(constants.%Copy.facet.14d, constants.%int_42.daa) [concrete = constants.%Inner.Get.specific_fn]
// CHECK:STDOUT:   %Inner.Get.call: init %i32 = call %Inner.Get.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Inner.Get.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f49) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f49
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1f5
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ac8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f49, constants.%U.51d) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%U.51d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c0fdca.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.5c5
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.48d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%T.f49, constants.%U.51d) {
// CHECK:STDOUT:   %T => constants.%T.f49
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_17.2 => constants.%.dedae6.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c0fdca.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.14d) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.14d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.535
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.f35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.14d, constants.%int_42.daa) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%i32
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%int_42.daa
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.104
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.fcb
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.cd1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%Copy.facet.14d, constants.%int_42.daa) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.14d
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_17.2 => constants.%.7ea
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.104
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %U => constants.%int_42.daa
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.7e2
// CHECK:STDOUT:   %.loc7_28 => constants.%.604
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.1a4
// CHECK:STDOUT:   %impl.elem0.loc7_28.2 => constants.%Int.as.Copy.impl.Op.ce7
// CHECK:STDOUT:   %bound_method.loc7_28.3 => constants.%Int.as.Copy.impl.Op.bound
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %bound_method.loc7_28.4 => constants.%bound_method.ece
// CHECK:STDOUT: }
// CHECK:STDOUT:
