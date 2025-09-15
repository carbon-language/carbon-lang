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
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.0f8: type = class_type @Outer, @Outer(%T.578) [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %U.57a: %T.as_type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Inner.type.0f1: type = generic_class_type @Inner, @Outer(%T.578) [symbolic]
// CHECK:STDOUT:   %Inner.generic.bb8: %Inner.type.0f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.b59: type = class_type @Inner, @Inner(%T.578, %U.57a) [symbolic]
// CHECK:STDOUT:   %Inner.Get.type.2e3: type = fn_type @Inner.Get, @Inner(%T.578, %U.57a) [symbolic]
// CHECK:STDOUT:   %Inner.Get.57b: %Inner.Get.type.2e3 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.022: <witness> = impl_witness @Inner.%Destroy.impl_witness_table, @Inner.as.Destroy.impl(%T.578, %U.57a) [symbolic]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %ptr.ce1: type = ptr_type %Inner.b59 [symbolic]
// CHECK:STDOUT:   %pattern_type.bec: type = pattern_type %ptr.ce1 [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.1e4: <witness> = impl_witness @Outer.%Destroy.impl_witness_table, @Outer.as.Destroy.impl(%T.578) [symbolic]
// CHECK:STDOUT:   %ptr.b2c: type = ptr_type %Outer.0f8 [symbolic]
// CHECK:STDOUT:   %pattern_type.3bd: type = pattern_type %ptr.b2c [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %bound_method.224: <bound method> = bound_method %U.57a, %impl.elem0.56e [symbolic]
// CHECK:STDOUT:   %bound_method.077: <bound method> = bound_method %U.57a, %specific_impl_fn.fb7 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %Outer.5dd: type = class_type @Outer, @Outer(%Copy.facet.26d) [concrete]
// CHECK:STDOUT:   %Inner.type.5cb: type = generic_class_type @Inner, @Outer(%Copy.facet.26d) [concrete]
// CHECK:STDOUT:   %Inner.generic.789: %Inner.type.5cb = struct_value () [concrete]
// CHECK:STDOUT:   %int_42.20e: Core.IntLiteral = int_value 42 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d46: <witness> = impl_witness imports.%Copy.impl_witness_table.462 [concrete]
// CHECK:STDOUT:   %Copy.facet.66a: %Copy.type = facet_value Core.IntLiteral, (%Copy.impl_witness.d46) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.e8c: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f01: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.acc: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.b6b, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.592: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.9ec = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.e8c = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.acc) [concrete]
// CHECK:STDOUT:   %.7ea: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.592 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.592, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.0fc: <bound method> = bound_method %int_42.20e, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_42.c68: %i32 = int_value 42 [concrete]
// CHECK:STDOUT:   %Inner.150: type = class_type @Inner, @Inner(%Copy.facet.26d, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Inner.Get.type.e7e: type = fn_type @Inner.Get, @Inner(%Copy.facet.26d, %int_42.c68) [concrete]
// CHECK:STDOUT:   %Inner.Get.43d: %Inner.Get.type.e7e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Inner.Get.43d, @Inner.Get(%Copy.facet.26d, %int_42.c68) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %int_42.c68, %Int.as.Copy.impl.Op.87e [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.6bf: <bound method> = bound_method %int_42.c68, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.c4d = import_ref Core//prelude/parts/copy, loc32_31, unloaded
// CHECK:STDOUT:   %Copy.impl_witness_table.462 = impl_witness_table (%Core.import_ref.c4d), @Core.IntLiteral.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.428: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.49e) = import_ref Core//prelude/parts/int, loc23_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f01)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.b6b = impl_witness_table (%Core.import_ref.428), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt [concrete]
// CHECK:STDOUT:   %.loc13: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var [concrete = %n.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Inner.as.Destroy.impl(@Outer.%T.loc4_13.2: %Copy.type, @Inner.%U.loc5_15.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Inner.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Inner.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Inner.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @Outer.as.Destroy.impl(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: @Outer.%Self.ref as constants.%Destroy.type {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %Outer.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @Outer.%Destroy.impl_witness
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     .Self = constants.%Outer.0f8
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: %Copy.type, %U.loc5_15.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type: type = fn_type @Inner.Get, @Inner(%T, %U.loc5_15.1) [symbolic = %Inner.Get.type (constants.%Inner.Get.type.2e3)]
// CHECK:STDOUT:   %Inner.Get: @Inner.%Inner.Get.type (%Inner.Get.type.2e3) = struct_value () [symbolic = %Inner.Get (constants.%Inner.Get.57b)]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.Get.decl: @Inner.%Inner.Get.type (%Inner.Get.type.2e3) = fn_decl @Inner.Get [symbolic = @Inner.%Inner.Get (constants.%Inner.Get.57b)] {
// CHECK:STDOUT:       %return.patt: @Inner.Get.%pattern_type (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Inner.Get.%pattern_type (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: %Copy.type = name_ref T, @Outer.%T.loc4_13.2 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc7_17.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc7_17: type = converted %T.ref, %T.as_type.loc7_17.2 [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %return.param: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = out_param call_param0
// CHECK:STDOUT:       %return: ref @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.b59
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Get = %Inner.Get.decl
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.Get(@Outer.%T.loc4_13.2: %Copy.type, @Inner.%U.loc5_15.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   %T: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T.578)]
// CHECK:STDOUT:   %T.as_type.loc7_17.1: type = facet_access_type %T [symbolic = %T.as_type.loc7_17.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc7_17.1 [symbolic = %pattern_type (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = bind_symbolic_name U, 1 [symbolic = %U (constants.%U.57a)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc7_17.1 [symbolic = %require_complete (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc7_17.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc7_28.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc7_28.2 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc7_28.2: @Inner.Get.%.loc7_28.2 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %bound_method.loc7_28.3: <bound method> = bound_method %U, %impl.elem0.loc7_28.2 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.224)]
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2: <specific function> = specific_impl_function %impl.elem0.loc7_28.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:   %bound_method.loc7_28.4: <bound method> = bound_method %U, %specific_impl_fn.loc7_28.2 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.077)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %U.ref: @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = name_ref U, @Inner.%U.loc5_15.2 [symbolic = %U (constants.%U.57a)]
// CHECK:STDOUT:     %impl.elem0.loc7_28.1: @Inner.Get.%.loc7_28.2 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc7_28.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc7_28.1: <bound method> = bound_method %U.ref, %impl.elem0.loc7_28.1 [symbolic = %bound_method.loc7_28.3 (constants.%bound_method.224)]
// CHECK:STDOUT:     %specific_impl_fn.loc7_28.1: <specific function> = specific_impl_function %impl.elem0.loc7_28.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc7_28.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc7_28.2: <bound method> = bound_method %U.ref, %specific_impl_fn.loc7_28.1 [symbolic = %bound_method.loc7_28.4 (constants.%bound_method.077)]
// CHECK:STDOUT:     %.loc7_28.1: init @Inner.Get.%T.as_type.loc7_17.1 (%T.as_type) = call %bound_method.loc7_28.2(%U.ref)
// CHECK:STDOUT:     return %.loc7_28.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Inner.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: %Copy.type, @Inner.%U.loc5_15.2: @Inner.%T.as_type.loc5_19.1 (%T.as_type)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Inner.as.Destroy.impl.Op.%ptr (%ptr.ce1)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Outer.as.Destroy.impl.Op(@Outer.%T.loc4_13.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @Outer.as.Destroy.impl.Op.%ptr (%ptr.b2c)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Copy.facet.loc13_23: %Copy.type = facet_value constants.%i32, (constants.%Copy.impl_witness.f0b) [concrete = constants.%Copy.facet.26d]
// CHECK:STDOUT:   %.loc13_23: %Copy.type = converted %i32, %Copy.facet.loc13_23 [concrete = constants.%Copy.facet.26d]
// CHECK:STDOUT:   %Outer: type = class_type @Outer, @Outer(constants.%Copy.facet.26d) [concrete = constants.%Outer.5dd]
// CHECK:STDOUT:   %.loc13_24: %Inner.type.5cb = specific_constant @Outer.%Inner.decl, @Outer(constants.%Copy.facet.26d) [concrete = constants.%Inner.generic.789]
// CHECK:STDOUT:   %Inner.ref: %Inner.type.5cb = name_ref Inner, %.loc13_24 [concrete = constants.%Inner.generic.789]
// CHECK:STDOUT:   %int_42: Core.IntLiteral = int_value 42 [concrete = constants.%int_42.20e]
// CHECK:STDOUT:   %Copy.facet.loc13_33: %Copy.type = facet_value Core.IntLiteral, (constants.%Copy.impl_witness.d46) [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:   %.loc13_33.1: %Copy.type = converted Core.IntLiteral, %Copy.facet.loc13_33 [concrete = constants.%Copy.facet.66a]
// CHECK:STDOUT:   %impl.elem0: %.7ea = impl_witness_access constants.%ImplicitAs.impl_witness.acc, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.592]
// CHECK:STDOUT:   %bound_method.loc13_33.1: <bound method> = bound_method constants.%int_42.20e, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc13_33.2: <bound method> = bound_method constants.%int_42.20e, %specific_fn [concrete = constants.%bound_method.0fc]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc13_33.2(constants.%int_42.20e) [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc13_33.2: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %.loc13_33.3: %i32 = converted constants.%int_42.20e, %.loc13_33.2 [concrete = constants.%int_42.c68]
// CHECK:STDOUT:   %Inner: type = class_type @Inner, @Inner(constants.%Copy.facet.26d, constants.%int_42.c68) [concrete = constants.%Inner.150]
// CHECK:STDOUT:   %.loc13_34: %Inner.Get.type.e7e = specific_constant @Inner.%Inner.Get.decl, @Inner(constants.%Copy.facet.26d, constants.%int_42.c68) [concrete = constants.%Inner.Get.43d]
// CHECK:STDOUT:   %Get.ref: %Inner.Get.type.e7e = name_ref Get, %.loc13_34 [concrete = constants.%Inner.Get.43d]
// CHECK:STDOUT:   %Inner.Get.specific_fn: <specific function> = specific_function %Get.ref, @Inner.Get(constants.%Copy.facet.26d, constants.%int_42.c68) [concrete = constants.%Inner.Get.specific_fn]
// CHECK:STDOUT:   %Inner.Get.call: init %i32 = call %Inner.Get.specific_fn()
// CHECK:STDOUT:   assign file.%n.var, %Inner.Get.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T.578) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T.578
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.0f1
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.bb8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T.578, constants.%U.57a) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%T.as_type
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%U.57a
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.2e3
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.57b
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ecc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%T.578, constants.%U.57a) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl(constants.%T.578, constants.%U.57a) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %U => constants.%U.57a
// CHECK:STDOUT:   %Inner => constants.%Inner.b59
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.022
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.as.Destroy.impl.Op(constants.%T.578, constants.%U.57a) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %T.as_type => constants.%T.as_type
// CHECK:STDOUT:   %U => constants.%U.57a
// CHECK:STDOUT:   %Inner => constants.%Inner.b59
// CHECK:STDOUT:   %ptr => constants.%ptr.ce1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.bec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Outer => constants.%Outer.0f8
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness.1e4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer.as.Destroy.impl.Op(constants.%T.578) {
// CHECK:STDOUT:   %T => constants.%T.578
// CHECK:STDOUT:   %Outer => constants.%Outer.0f8
// CHECK:STDOUT:   %ptr => constants.%ptr.b2c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3bd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%Copy.facet.26d) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%Copy.facet.26d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.5cb
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.789
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%Copy.facet.26d, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.26d
// CHECK:STDOUT:   %T.as_type.loc5_19.1 => constants.%i32
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%int_42.c68
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.Get.type => constants.%Inner.Get.type.e7e
// CHECK:STDOUT:   %Inner.Get => constants.%Inner.Get.43d
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner.Get(constants.%Copy.facet.26d, constants.%int_42.c68) {
// CHECK:STDOUT:   %T => constants.%Copy.facet.26d
// CHECK:STDOUT:   %T.as_type.loc7_17.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %U => constants.%int_42.c68
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.f0b
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.26d
// CHECK:STDOUT:   %.loc7_28.2 => constants.%.3c4
// CHECK:STDOUT:   %impl.elem0.loc7_28.2 => constants.%Int.as.Copy.impl.Op.87e
// CHECK:STDOUT:   %bound_method.loc7_28.3 => constants.%Int.as.Copy.impl.Op.bound
// CHECK:STDOUT:   %specific_impl_fn.loc7_28.2 => constants.%Int.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT:   %bound_method.loc7_28.4 => constants.%bound_method.6bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
