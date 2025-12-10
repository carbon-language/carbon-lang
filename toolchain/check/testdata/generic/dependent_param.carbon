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
// CHECK:STDOUT:   %T.f92: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.44e: type = class_type @Outer, @Outer(%T.f92) [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.f92 [symbolic]
// CHECK:STDOUT:   %U.9fd: %T.binding.as_type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c769df.1: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Inner.type.561: type = generic_class_type @Inner, @Outer(%T.f92) [symbolic]
// CHECK:STDOUT:   %Inner.generic.3b5: %Inner.type.561 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.485: type = class_type @Inner, @Inner(%T.f92, %U.9fd) [symbolic]
// CHECK:STDOUT:   %Inner.Get.type.a07: type = fn_type @Inner.Get, @Inner(%T.f92, %U.9fd) [symbolic]
// CHECK:STDOUT:   %Inner.Get.15d: %Inner.Get.type.a07 = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.91e: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.edd: <witness> = lookup_impl_witness %T.f92, @Copy [symbolic]
// CHECK:STDOUT:   %.232: type = fn_type_with_self_type %Copy.Op.type, %T.f92 [symbolic]
// CHECK:STDOUT:   %impl.elem0.8df: %.232 = impl_witness_access %Copy.lookup_impl_witness.edd, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.5c4: <specific function> = specific_impl_function %impl.elem0.8df, @Copy.Op(%T.f92) [symbolic]
// CHECK:STDOUT:   %bound_method.293: <bound method> = bound_method %U.9fd, %impl.elem0.8df [symbolic]
// CHECK:STDOUT:   %bound_method.ed9: <bound method> = bound_method %U.9fd, %specific_impl_fn.5c4 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.413: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.2d6: %Int.as.Copy.impl.Op.type.413 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.09c: <witness> = impl_witness imports.%Copy.impl_witness_table.e1c, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.60b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.60b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.6df: %Copy.type = facet_value %i32, (%Copy.impl_witness.09c) [concrete]
// CHECK:STDOUT:   %Outer.732: type = class_type @Outer, @Outer(%Copy.facet.6df) [concrete]
// CHECK:STDOUT:   %Inner.type.de0: type = generic_class_type @Inner, @Outer(%Copy.facet.6df) [concrete]
// CHECK:STDOUT:   %Inner.generic.6fc: %Inner.type.de0 = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d08: <witness> = impl_witness imports.%Copy.impl_witness_table.5a1 [concrete]
// CHECK:STDOUT:   %Copy.facet.8aa: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.d08) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.ec0: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.99e, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.ec0) [concrete]
// CHECK:STDOUT:   %.d28: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.5bc: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Inner.81e: type = class_type @Inner, @Inner(%Copy.facet.6df, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Inner.Get.type.071: type = fn_type @Inner.Get, @Inner(%Copy.facet.6df, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Inner.Get.0d9: %Inner.Get.type.071 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Inner.Get.0d9, @Inner.Get(%Copy.facet.6df, %int_42.c68) [concrete]
// CHECK:STDOUT:   %.fe5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.6df [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_42.c68, %Int.as.Copy.impl.Op.c85 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c85, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.436: <bound method> = bound_method %int_42.c68, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.edf: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.413) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.2d6)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e1c = impl_witness_table (%Core.import_ref.edf), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.b02 = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.5a1 = impl_witness_table (%Core.import_ref.b02), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.3e1: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.99e = impl_witness_table (%Core.import_ref.3e1), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = ref_binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt [concrete]
// CHECK:STDOUT:   %.loc13: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = ref_binding n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: %Copy.type) {
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
// CHECK:STDOUT:     .Self = constants.%Outer.44e
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: %Copy.type, %U.loc5_15.2: @Inner.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type: type = fn_type @Inner.Get, @Inner(%T, %U.loc5_15.1) [symbolic = %Inner.Get.type (constants.%Inner.Get.type.a07)]
// CHECK:STDOUT:   %Inner.Get: @Inner.%Inner.Get.type (%Inner.Get.type.a07) = struct_value () [symbolic = %Inner.Get (constants.%Inner.Get.15d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.Get.decl: @Inner.%Inner.Get.type (%Inner.Get.type.a07) = fn_decl @Inner.Get [symbolic = @Inner.%Inner.Get (constants.%Inner.Get.15d)] {
// CHECK:STDOUT:       %return.patt: @Inner.Get.%pattern_type (%pattern_type.c769df.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.Get.%pattern_type (%pattern_type.c769df.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.f92)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc7_17: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %return.param: ref @Inner.Get.%T.binding.as_type (%T.binding.as_type) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.Get.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.485
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Inner.Get.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.Get(@Outer.%T.loc4_13.2: %Copy.type, @Inner.%U.loc5_15.2: @Inner.%T.binding.as_type (%T.binding.as_type)) {
// CHECK:STDOUT:   %T: %Copy.type = symbolic_binding T, 0 [symbolic = %T (constants.%T.f92)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.c769df.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.91e)]
// CHECK:STDOUT:   %U: @Inner.Get.%T.binding.as_type (%T.binding.as_type) = symbolic_binding U, 1 [symbolic = %U (constants.%U.9fd)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.edd)]
// CHECK:STDOUT:   %.loc7_28.2: type = fn_type_with_self_type constants.%Copy.Op.type, %T [symbolic = %.loc7_28.2 (constants.%.232)]
// CHECK:STDOUT:   %impl.elem0.loc7_28.2: @Inner.Get.%.loc7_28.2 (%.232) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.8df)]
// CHECK:STDOUT:   %bound_method.loc7_28.3: <bound method> = bound_method %U, %impl.elem0.loc7_28.2 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.293)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2: <specific function> = specific_impl_function %impl.elem0.loc7_28.2, @Copy.Op(%T) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.5c4)]
// CHECK:STDOUT:   %bound_method.loc7_28.4: <bound method> = bound_method %U, %specific_impl_fn.loc7_28.2 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.ed9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> %return.param: @Inner.Get.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @Inner.Get.%T.binding.as_type (%T.binding.as_type) = name_ref U, @Inner.%U.loc5_15.2 [symbolic = %U (constants.%U.9fd)]
// CHECK:STDOUT:     %impl.elem0.loc7_28.1: @Inner.Get.%.loc7_28.2 (%.232) = impl_witness_access constants.%Copy.lookup_impl_witness.edd, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.8df)]
// CHECK:STDOUT:     %bound_method.loc7_28.1: <bound method> = bound_method %U.ref, %impl.elem0.loc7_28.1 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.293)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_28.1: <specific function> = specific_impl_function %impl.elem0.loc7_28.1, @Copy.Op(constants.%T.f92) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.5c4)]
// CHECK:STDOUT:     %bound_method.loc7_28.2: <bound method> = bound_method %U.ref, %specific_impl_fn.loc7_28.1 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.ed9)]
// CHECK:STDOUT:     %.loc7_14: ref @Inner.Get.%T.binding.as_type (%T.binding.as_type) = splice_block %return {}
// CHECK:STDOUT:     %.loc7_28.1: init @Inner.Get.%T.binding.as_type (%T.binding.as_type) = call %bound_method.loc7_28.2(%U.ref) to %.loc7_14
// CHECK:STDOUT:     return %.loc7_28.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_23: %Copy.type = facet_value %i32, (constants.%Copy.impl_witness.09c) [concrete = constants.%Copy.facet.6df]
// CHECK:STDOUT:   %.loc13_23: %Copy.type = converted %i32, %Copy.facet.loc13_23 [concrete = constants.%Copy.facet.6df]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%Copy.facet.6df) [concrete = constants.%Outer.732]
// CHECK:STDOUT:   %.loc13_24: %Inner.type.de0 = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.6df) [concrete = constants.%Inner.generic.6fc]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.de0 = name_ref Inner, %.loc13_24 [concrete = constants.%Inner.generic.6fc]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %Copy.facet.loc13_33: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d08) [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %.loc13_33.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_33 [concrete = constants.%Copy.facet.8aa]
// CHECK:STDOUT:   %impl.elem0: %.d28 = impl_witness_access constants.%ImplicitAs.impl_witness.ec0, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc13_33.1: <bound method> = bound_method constants.%int_42.20e, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_33.2: <bound method> = bound_method constants.%int_42.20e, %specific_fn [concrete = constants.%bound_method.5bc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_33.2(constants.%int_42.20e) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc13_33.3: %i32 = converted constants.%int_42.20e, %.loc13_33.2 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%Copy.facet.6df, constants.%int_42.c68) [concrete = constants.%Inner.81e]
// CHECK:STDOUT:   %.loc13_34: %Inner.Get.type.071 = specific_constant @Inner.%Inner.Get.decl, @Inner(constants.%Copy.facet.6df, constants.%int_42.c68) [concrete = constants.%Inner.Get.0d9]
// CHECK:STDOUT:   %Get.ref: %Inner.Get.type.071 = name_ref Get, %.loc13_34 [concrete = constants.%Inner.Get.0d9]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Get.ref, @Inner.Get(constants.%Copy.facet.6df, constants.%int_42.c68) [concrete = constants.%Inner.Get.specific_fn]
// CHECK:STDOUT:   %Inner.Get.call: init %i32 = call %Inner.Get.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Inner.Get.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.f92) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.f92
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.561
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.3b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.f92, constants.%U.9fd) {
// CHECK:STDOUT:   %T => constants.%T.f92
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%U.9fd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c769df.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.a07
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.15d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%T.f92, constants.%U.9fd) {
// CHECK:STDOUT:   %T => constants.%T.f92
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.c769df.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.6df) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.6df
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.de0
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.6fc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.6df, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.6df
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%int_42.c68
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.071
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.0d9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%Copy.facet.6df, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.6df
// CHECK:STDOUT:   %T.binding.as_type => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %U => constants.%int_42.c68
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.09c
// CHECK:STDOUT:   %.loc7_28.2 => constants.%.fe5
// CHECK:STDOUT:   %impl.elem0.loc7_28.2 => constants.%Int.as.Copy.impl.Op.c85
// CHECK:STDOUT:   %bound_method.loc7_28.3 => constants.%Int.as.Copy.impl.Op.bound
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %bound_method.loc7_28.4 => constants.%bound_method.436
// CHECK:STDOUT: }
// CHECK:STDOUT:
