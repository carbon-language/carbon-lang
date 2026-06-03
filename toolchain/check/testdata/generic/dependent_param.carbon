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
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.409: type = class_type @Outer, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %U.7e5: %T.as_type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.272: type = generic_class_type @Inner, @Outer(%T.f84) [symbolic]
// CHECK:STDOUT:   %Inner.generic.e21: %Inner.type.272 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.698: type = class_type @Inner, @Inner(%T.f84, %U.7e5) [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.Get.type.8de: type = fn_type @Inner.Get, @Inner(%T.f84, %U.7e5) [symbolic]
// CHECK:STDOUT:   %Inner.Get.e2c: %Inner.Get.type.8de = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %bound_method.023: <bound method> = bound_method %U.7e5, %impl.elem0.bd2 [symbolic]
// CHECK:STDOUT:   %bound_method.0b2: <bound method> = bound_method %U.7e5, %specific_impl_fn.e2a [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.987: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Outer.90e: type = class_type @Outer, @Outer(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %Inner.type.053: type = generic_class_type @Inner, @Outer(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %Inner.generic.ab4: %Inner.type.053 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.6da: <witness> = impl_witness imports.%Copy.impl_witness_table.06e [concrete]
// CHECK:STDOUT:   %Copy.facet.f12: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.6da) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.914: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.845: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.a23: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b3c, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.2ba = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.914 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.a23) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.740: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.1c5: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.740, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e39, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.098: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.ac4: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Inner.aad: type = class_type @Inner, @Inner(%Copy.facet.987, %int_42.ac4) [concrete]
// CHECK:STDOUT:   %Inner.Get.type.6e9: type = fn_type @Inner.Get, @Inner(%Copy.facet.987, %int_42.ac4) [concrete]
// CHECK:STDOUT:   %Inner.Get.bb6: %Inner.Get.type.6e9 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Inner.Get.bb6, @Inner.Get(%Copy.facet.987, %int_42.ac4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet.987 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_42.ac4, %Int.as.Copy.impl.Op.4f6 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.590: <bound method> = bound_method %int_42.ac4, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.bbf = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.06e = impl_witness_table (%Core.import_ref.bbf), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.edf: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.e74) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.845)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b3c = impl_witness_table (%Core.import_ref.edf), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.6b6 = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.6b6 = var_pattern %n.patt [concrete]
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
// CHECK:STDOUT:     .Self = constants.%Outer.409
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: %Copy.type, %U.loc5_16.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type: type = fn_type @Inner.Get, @Inner(%T, %U.loc5_16.1) [symbolic = %Inner.Get.type (constants.%Inner.Get.type.8de)]
// CHECK:STDOUT:   %Inner.Get: @Inner.%Inner.Get.type (%Inner.Get.type.8de) = struct_value () [symbolic = %Inner.Get (constants.%Inner.Get.e2c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.Get.decl: @Inner.%Inner.Get.type (%Inner.Get.type.8de) = fn_decl @Inner.Get [symbolic = @Inner.%Inner.Get (constants.%Inner.Get.e2c)] {
// CHECK:STDOUT:       %return.param_patt: @Inner.Get.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @Inner.Get.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt, %.loc7_17.3 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_14.2 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc7_17.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_17.3: type = converted %T.ref, %T.as_type.loc7_17.2 [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_17.4: Core.Form = init_form %.loc7_17.3 [symbolic = %.loc7_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:       %return.param: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.698
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Inner.Get.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.Get(@Outer.%T.loc4_14.2: %Copy.type, @Inner.%U.loc5_16.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc7_17.1: type = facet_access_type %T [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %.loc7_17.2: Core.Form = init_form %T.as_type.loc7_17.1 [symbolic = %.loc7_17.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_17.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_17.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %U: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = symbolic_binding U, 1 [symbolic = %U (constants.%U.7e5)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc7_28: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T [symbolic = %.loc7_28 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc7_28.2: @Inner.Get.%.loc7_28 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %bound_method.loc7_28.3: <bound method> = bound_method %U, %impl.elem0.loc7_28.2 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.023)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2: <specific function> = specific_impl_function %impl.elem0.loc7_28.2, @Copy.WithSelf.Op(%T) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:   %bound_method.loc7_28.4: <bound method> = bound_method %U, %specific_impl_fn.loc7_28.2 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.0b2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = name_ref U, @Inner.%U.loc5_16.2 [symbolic = %U (constants.%U.7e5)]
// CHECK:STDOUT:     %impl.elem0.loc7_28.1: @Inner.Get.%.loc7_28 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc7_28.1: <bound method> = bound_method %U.ref, %impl.elem0.loc7_28.1 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.023)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_28.1: <specific function> = specific_impl_function %impl.elem0.loc7_28.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc7_28.2: <bound method> = bound_method %U.ref, %specific_impl_fn.loc7_28.1 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.0b2)]
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
// CHECK:STDOUT:   %Copy.facet.loc13_23: %Copy.type = facet_value %i32, (constants.%Copy.impl_witness.0e0) [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:   %.loc13_23: %Copy.type = converted %i32, %Copy.facet.loc13_23 [concrete = constants.%Copy.facet.987]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%Copy.facet.987) [concrete = constants.%Outer.90e]
// CHECK:STDOUT:   %.loc13_24: %Inner.type.053 = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.987) [concrete = constants.%Inner.generic.ab4]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.053 = name_ref Inner, %.loc13_24 [concrete = constants.%Inner.generic.ab4]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %Copy.facet.loc13_33: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.6da) [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %.loc13_33.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_33 [concrete = constants.%Copy.facet.f12]
// CHECK:STDOUT:   %impl.elem0: %.1c5 = impl_witness_access constants.%ImplicitAs.impl_witness.a23, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e39]
// CHECK:STDOUT:   %bound_method.loc13_33.1: <bound method> = bound_method constants.%int_42.20e, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_33.2: <bound method> = bound_method constants.%int_42.20e, %specific_fn [concrete = constants.%bound_method.098]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_33.2(constants.%int_42.20e) [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %.loc13_33.3: %i32 = converted constants.%int_42.20e, %.loc13_33.2 [concrete = constants.%int_42.ac4]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%Copy.facet.987, constants.%int_42.ac4) [concrete = constants.%Inner.aad]
// CHECK:STDOUT:   %.loc13_34: %Inner.Get.type.6e9 = specific_constant @Inner.%Inner.Get.decl, @Inner(constants.%Copy.facet.987, constants.%int_42.ac4) [concrete = constants.%Inner.Get.bb6]
// CHECK:STDOUT:   %Get.ref: %Inner.Get.type.6e9 = name_ref Get, %.loc13_34 [concrete = constants.%Inner.Get.bb6]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Get.ref, @Inner.Get(constants.%Copy.facet.987, constants.%int_42.ac4) [concrete = constants.%Inner.Get.specific_fn]
// CHECK:STDOUT:   %Inner.Get.call: init %i32 = call %Inner.Get.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Inner.Get.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T.f84
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.272
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.e21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f84, constants.%U.7e5) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%U.7e5
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.8de
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.e2c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%T.f84, constants.%U.7e5) {
// CHECK:STDOUT:   %T => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %.loc7_17.2 => constants.%.1ce700.2
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.987) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%Copy.facet.987
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.053
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.ab4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.987, constants.%int_42.ac4) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.987
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%i32
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%int_42.ac4
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.6e9
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.bb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%Copy.facet.987, constants.%int_42.ac4) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.987
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%i32
// CHECK:STDOUT:   %.loc7_17.2 => constants.%.795
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6b6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %U => constants.%int_42.ac4
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d09
// CHECK:STDOUT:   %.loc7_28 => constants.%.7a6
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.0e0
// CHECK:STDOUT:   %impl.elem0.loc7_28.2 => constants.%Int.as.Copy.impl.Op.4f6
// CHECK:STDOUT:   %bound_method.loc7_28.3 => constants.%Int.as.Copy.impl.Op.bound
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %bound_method.loc7_28.4 => constants.%bound_method.590
// CHECK:STDOUT: }
// CHECK:STDOUT:
