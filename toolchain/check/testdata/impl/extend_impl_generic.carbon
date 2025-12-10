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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/impl/extend_impl_generic.carbon

// --- extend_impl_generic_interface.carbon

library "[[@TEST_NAME]]";

interface HasF(T:! type) {
  fn F() -> T;
}

class Param {
  var x: i32;
}

class C {
  extend impl as HasF(Param) {
    fn F() -> Param {
      return {.x = 2};
    }
  }
}

fn G(c: C) {
  let a: i32 = C.F().x;
  var b: i32 = c.F().x;
}

// --- extend_impl_generic_class.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F[self: Self](t: T);
}

class X(U:! type) {
  extend impl as I(U) {
    fn F[self: Self](t: U) { }
  }
}

// CHECK:STDOUT: --- extend_impl_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %HasF.type.ee1: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.ee1 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.4b7: type = facet_type <@HasF, @HasF(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.c5a: %HasF.type.4b7 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %HasF.F.type.607: type = fn_type @HasF.F, @HasF(%T.67d) [symbolic]
// CHECK:STDOUT:   %HasF.F.39f: %HasF.F.type.607 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.196: type = assoc_entity_type @HasF, @HasF(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.303: %HasF.assoc_type.196 = assoc_entity element0, @HasF.%HasF.F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type.31f: type = facet_type <@HasF, @HasF(%Param)> [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.e81: %HasF.type.31f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.F.type.f31: type = fn_type @HasF.F, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %HasF.F.85a: %HasF.F.type.f31 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.078: type = assoc_entity_type @HasF, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %assoc0.2b0: %HasF.assoc_type.078 = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %pattern_type.9ea: type = pattern_type %Param [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.31f = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.c96 = struct_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.740: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57f: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.8c3, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.84b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.57f) [concrete]
// CHECK:STDOUT:   %.dbf: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Param.val: %Param = struct_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.7c7: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.63a: type = fn_type_with_self_type %HasF.F.type.f31, %HasF.facet [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.413: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.2d6: %Int.as.Copy.impl.Op.type.413 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.66f: <witness> = impl_witness imports.%Copy.impl_witness_table.373, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.60b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.60b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.66f) [concrete]
// CHECK:STDOUT:   %.958: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c85, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value.ded: %type_where = facet_value %Param, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d0d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.ded) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.960: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.d0d = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4a0: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.960, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.ded) [concrete]
// CHECK:STDOUT:   %facet_value.d23: %type_where = facet_value %i32, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value.d23) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.ae2: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.9df = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.351: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value.d23) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.1b5: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.412) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.740)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.8c3 = impl_witness_table (%Core.import_ref.1b5), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.30b: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.413) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.2d6)]
// CHECK:STDOUT:   %Copy.impl_witness_table.373 = impl_witness_table (%Core.import_ref.30b), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .HasF = %HasF.decl
// CHECK:STDOUT:     .Param = %Param.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %HasF.decl: %HasF.type.ee1 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_16.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %pattern_type.7c7 = value_binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.7c7 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_16.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.1)> [symbolic = %HasF.type (constants.%HasF.type.4b7)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.4b7) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.c5a)]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F, @HasF(%T.loc4_16.1) [symbolic = %HasF.F.type (constants.%HasF.F.type.607)]
// CHECK:STDOUT:   %HasF.F: @HasF.%HasF.F.type (%HasF.F.type.607) = struct_value () [symbolic = %HasF.F (constants.%HasF.F.39f)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF, @HasF(%T.loc4_16.1) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.196)]
// CHECK:STDOUT:   %assoc0.loc5_14.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.196) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.303)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.4b7) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.c5a)]
// CHECK:STDOUT:     %HasF.F.decl: @HasF.%HasF.F.type (%HasF.F.type.607) = fn_decl @HasF.F [symbolic = @HasF.%HasF.F (constants.%HasF.F.39f)] {
// CHECK:STDOUT:       %return.patt: @HasF.F.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @HasF.F.%pattern_type (%pattern_type.51d1c4.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %return.param: ref @HasF.F.%T (%T.67d) = out_param call_param0
// CHECK:STDOUT:       %return: ref @HasF.F.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.196) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.303)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_14.1
// CHECK:STDOUT:     .Param = <poisoned>
// CHECK:STDOUT:     witness = (%HasF.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.9ea = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ea = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param call_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = %HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @C.as.HasF.impl [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.ee1 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [concrete = constants.%HasF.type.31f]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @C.as.HasF.impl.%HasF.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%T.loc4_16.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.4b7)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @HasF.F.%T (%T.67d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() -> %return.param: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_21.1: %struct_type.x.c96 = struct_literal (%int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.dbf = impl_witness_access constants.%ImplicitAs.impl_witness.57f, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f0]
// CHECK:STDOUT:   %bound_method.loc15_21.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_21.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_21.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.2 to %.loc15_21.3 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.5: init %Param = class_init (%.loc15_21.4), %return [concrete = constants.%Param.val]
// CHECK:STDOUT:   %.loc15_22: init %Param = converted %.loc15_21.1, %.loc15_21.5 [concrete = constants.%Param.val]
// CHECK:STDOUT:   return %.loc15_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc21_17: %HasF.assoc_type.078 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.2b0]
// CHECK:STDOUT:   %F.ref.loc21: %HasF.assoc_type.078 = name_ref F, %.loc21_17 [concrete = constants.%assoc0.2b0]
// CHECK:STDOUT:   %impl.elem0.loc21: %.63a = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc21_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc21: init %Param = call %impl.elem0.loc21() to %.loc21_20.1
// CHECK:STDOUT:   %.loc21_20.2: ref %Param = temporary %.loc21_20.1, %C.as.HasF.impl.F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc21_21.1: ref %i32 = class_element_access %.loc21_20.2, element0
// CHECK:STDOUT:   %.loc21_10: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_21.2: %i32 = acquire_value %.loc21_21.1
// CHECK:STDOUT:   %a: %i32 = value_binding a, %.loc21_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_17: %HasF.assoc_type.078 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.2b0]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type.078 = name_ref F, %.loc22_17 [concrete = constants.%assoc0.2b0]
// CHECK:STDOUT:   %impl.elem0.loc22_17: %.63a = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc22_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc22: init %Param = call %impl.elem0.loc22_17() to %.loc22_20.1
// CHECK:STDOUT:   %.loc22_20.2: ref %Param = temporary %.loc22_20.1, %C.as.HasF.impl.F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc22_21.1: ref %i32 = class_element_access %.loc22_20.2, element0
// CHECK:STDOUT:   %.loc22_21.2: %i32 = acquire_value %.loc22_21.1
// CHECK:STDOUT:   %impl.elem0.loc22_21: %.958 = impl_witness_access constants.%Copy.impl_witness.66f, element0 [concrete = constants.%Int.as.Copy.impl.Op.c85]
// CHECK:STDOUT:   %bound_method.loc22_21.1: <bound method> = bound_method %.loc22_21.2, %impl.elem0.loc22_21
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc22_21, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_21.2: <bound method> = bound_method %.loc22_21.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_21.2(%.loc22_21.2)
// CHECK:STDOUT:   assign %b.var, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %.loc22_10: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22_20: <bound method> = bound_method %.loc22_20.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.960
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.960, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ded) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4a0]
// CHECK:STDOUT:   %bound_method.loc22_20: <bound method> = bound_method %.loc22_20.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.1
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22_20: init %empty_tuple.type = call %bound_method.loc22_20(%.loc22_20.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc22_3: <bound method> = bound_method %b.var, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.ae2, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.d23) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.351]
// CHECK:STDOUT:   %bound_method.loc22_3: <bound method> = bound_method %b.var, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.2
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc22_3: init %empty_tuple.type = call %bound_method.loc22_3(%b.var)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %.loc21_20.2, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.960
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.960, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value.ded) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.4a0]
// CHECK:STDOUT:   %bound_method.loc21: <bound method> = bound_method %.loc21_20.2, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn.3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %bound_method.loc21(%.loc21_20.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T.67d, constants.%Self.c5a) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.31f
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.e81
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.f31
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.85a
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.078
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.2b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9ea
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.609: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.609 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.d71ca1.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.bb7324.1: %I.type.d71ca1.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self.bb7324.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.b51: type = pattern_type %Self.binding.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %I.F.type.93703d.1: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.a37974.1: %I.F.type.93703d.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c031.1: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.6a687a.1: %I.assoc_type.76c031.1 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.d71ca1.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.as.I.impl.%I.impl_witness_table, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %Self.bb7324.2: %I.type.d71ca1.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.93703d.2: type = fn_type @I.F, @I(%U) [symbolic]
// CHECK:STDOUT:   %I.F.a37974.2: %I.F.type.93703d.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.76c031.2: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.6a687a.2: %I.assoc_type.76c031.2 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.073: <witness> = require_complete_type %I.type.d71ca1.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.e07: type = pattern_type %X [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.d71ca1.2 = facet_value %X, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.5dd: <witness> = require_complete_type %X [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %U [symbolic]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.609 = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc8_9.2: type = symbolic_binding U, 0 [symbolic = %U.loc8_9.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.1)> [symbolic = %I.type (constants.%I.type.d71ca1.1)]
// CHECK:STDOUT:   %Self.loc4_23.2: @I.%I.type (%I.type.d71ca1.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.bb7324.1)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T.loc4_13.1) [symbolic = %I.F.type (constants.%I.F.type.93703d.1)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.93703d.1) = struct_value () [symbolic = %I.F (constants.%I.F.a37974.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.76c031.1)]
// CHECK:STDOUT:   %assoc0.loc5_25.2: @I.%I.assoc_type (%I.assoc_type.76c031.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.6a687a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @I.%I.type (%I.type.d71ca1.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.bb7324.1)]
// CHECK:STDOUT:     %I.F.decl: @I.%I.F.type (%I.F.type.93703d.1) = fn_decl @I.F [symbolic = @I.%I.F (constants.%I.F.a37974.1)] {
// CHECK:STDOUT:       %self.patt: @I.F.%pattern_type.loc5_8 (%pattern_type.b51) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @I.F.%pattern_type.loc5_8 (%pattern_type.b51) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @I.F.%pattern_type.loc5_20 (%pattern_type.51d1c4.1) = value_binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @I.F.%pattern_type.loc5_20 (%pattern_type.51d1c4.1) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @I.F.%I.type (%I.type.d71ca1.1) = specific_constant @I.%Self.loc4_23.1, @I(constants.%T) [symbolic = %Self (constants.%Self.bb7324.1)]
// CHECK:STDOUT:         %Self.ref: @I.F.%I.type (%I.type.d71ca1.1) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.bb7324.1)]
// CHECK:STDOUT:         %Self.as_type: type = facet_access_type %Self.ref [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.F.%Self.binding.as_type (%Self.binding.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @I.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @I.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.%I.assoc_type (%I.assoc_type.76c031.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.6a687a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_25.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     witness = (%I.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @X.as.I.impl(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.1: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.d71ca1.2)]
// CHECK:STDOUT:   %I.impl_witness.loc9_23.2: <witness> = impl_witness %I.impl_witness_table, @X.as.I.impl(%U) [symbolic = %I.impl_witness.loc9_23.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc9_21.1 [symbolic = %require_complete (constants.%require_complete.073)]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic = %X.as.I.impl.F.type (constants.%X.as.I.impl.F.type)]
// CHECK:STDOUT:   %X.as.I.impl.F: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = struct_value () [symbolic = %X.as.I.impl.F (constants.%X.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.2 {
// CHECK:STDOUT:     %X.as.I.impl.F.decl: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = fn_decl @X.as.I.impl.F [symbolic = @X.as.I.impl.%X.as.I.impl.F (constants.%X.as.I.impl.F)] {
// CHECK:STDOUT:       %self.patt: @X.as.I.impl.F.%pattern_type.loc10_10 (%pattern_type.e07) = value_binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @X.as.I.impl.F.%pattern_type.loc10_10 (%pattern_type.e07) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @X.as.I.impl.F.%pattern_type.loc10_22 (%pattern_type.51d1c4.2) = value_binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @X.as.I.impl.F.%pattern_type.loc10_22 (%pattern_type.51d1c4.2) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @X.as.I.impl.F.%X (%X) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc10_16.2: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_16.2 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @X.as.I.impl.F.%X (%X) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @X.as.I.impl.F.%U (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %t: @X.as.I.impl.F.%U (%U) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%X.as.I.impl.F.decl), @X.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc9_23.1: <witness> = impl_witness %I.impl_witness_table, @X.as.I.impl(constants.%U) [symbolic = %I.impl_witness.loc9_23.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %X.as.I.impl.F.decl
// CHECK:STDOUT:     witness = %I.impl_witness.loc9_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U.loc8_9.1: type = symbolic_binding U, 0 [symbolic = %U.loc8_9.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.1)> [symbolic = %I.type (constants.%I.type.d71ca1.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.073)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @X.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.609 = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.2: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.d71ca1.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: type = specific_constant @X.as.I.impl.%I.type.loc9_21.2, @X.as.I.impl(constants.%U) [symbolic = %I.type (constants.%I.type.d71ca1.2)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%X
// CHECK:STDOUT:     .I = <poisoned>
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     extend %.loc9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @I.F(@I.%T.loc4_13.2: type, @I.%Self.loc4_23.1: @I.%I.type (%I.type.d71ca1.1)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.d71ca1.1)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.d71ca1.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.bb7324.1)]
// CHECK:STDOUT:   %Self.binding.as_type: type = symbolic_binding_type Self, 1, %Self [symbolic = %Self.binding.as_type (constants.%Self.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.binding.as_type [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.b51)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %T [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.binding.as_type (%Self.binding.as_type), %t.param: @I.F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.as.I.impl.F(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type.loc10_10: type = pattern_type %X [symbolic = %pattern_type.loc10_10 (constants.%pattern_type.e07)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.d71ca1.2)]
// CHECK:STDOUT:   %require_complete.loc10_25: <witness> = require_complete_type %I.type [symbolic = %require_complete.loc10_25 (constants.%require_complete.073)]
// CHECK:STDOUT:   %pattern_type.loc10_22: type = pattern_type %U [symbolic = %pattern_type.loc10_22 (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_14: <witness> = require_complete_type %X [symbolic = %require_complete.loc10_14 (constants.%require_complete.5dd)]
// CHECK:STDOUT:   %require_complete.loc10_23: <witness> = require_complete_type %U [symbolic = %require_complete.loc10_23 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @X.as.I.impl.F.%X (%X), %t.param: @X.as.I.impl.F.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.bb7324.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.d71ca1.1
// CHECK:STDOUT:   %Self => constants.%Self.bb7324.1
// CHECK:STDOUT:   %Self.binding.as_type => constants.%Self.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.b51
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.d71ca1.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.073
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.d71ca1.2
// CHECK:STDOUT:   %Self.loc4_23.2 => constants.%Self.bb7324.2
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.93703d.2
// CHECK:STDOUT:   %I.F => constants.%I.F.a37974.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.76c031.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.6a687a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.1 => constants.%I.type.d71ca1.2
// CHECK:STDOUT:   %I.impl_witness.loc9_23.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.073
// CHECK:STDOUT:   %X.as.I.impl.F.type => constants.%X.as.I.impl.F.type
// CHECK:STDOUT:   %X.as.I.impl.F => constants.%X.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl.F(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type.loc10_10 => constants.%pattern_type.e07
// CHECK:STDOUT:   %I.type => constants.%I.type.d71ca1.2
// CHECK:STDOUT:   %require_complete.loc10_25 => constants.%require_complete.073
// CHECK:STDOUT:   %pattern_type.loc10_22 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.d71ca1.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.binding.as_type => constants.%X
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.e07
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
