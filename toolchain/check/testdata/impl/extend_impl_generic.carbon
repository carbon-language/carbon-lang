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
  let unused a: i32 = C.F().x;
  var unused b: i32 = c.F().x;
}

// --- extend_impl_generic_class.carbon

library "[[@TEST_NAME]]";

interface I(T:! type) {
  fn F[self: Self](t: T);
}

class X(U:! type) {
  extend impl as I(U) {
    fn F[unused self: Self](unused t: U) { }
  }
}

// CHECK:STDOUT: --- extend_impl_generic_interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.a6d: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.a6d = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.81d: type = facet_type <@HasF, @HasF(%T.67d)> [symbolic]
// CHECK:STDOUT:   %Self.fea: %HasF.type.81d = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %.184347.1: Core.Form = init_form %T.67d [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.cd0: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%T.67d, %Self.fea) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.252: %HasF.WithSelf.F.type.cd0 = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.ea5: type = assoc_entity_type @HasF, @HasF(%T.67d) [symbolic]
// CHECK:STDOUT:   %assoc0.702: %HasF.assoc_type.ea5 = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %pattern_type.c6b: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.2bb: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.fa5: <witness> = complete_type_witness %struct_type.x.2bb [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type.264: type = facet_type <@HasF, @HasF(%Param)> [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.as.HasF.impl.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.975: %HasF.type.264 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.7ea: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Param, %Self.fea) [symbolic]
// CHECK:STDOUT:   %HasF.WithSelf.F.1f9: %HasF.WithSelf.F.type.7ea = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.2be: type = assoc_entity_type @HasF, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %assoc0.b2a: %HasF.assoc_type.2be = assoc_entity element0, @HasF.WithSelf.%HasF.WithSelf.F.decl [concrete]
// CHECK:STDOUT:   %.d1b: Core.Form = init_form %Param [concrete]
// CHECK:STDOUT:   %pattern_type.8e1: type = pattern_type %Param [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.264 = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.fd8: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Param, %HasF.facet) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.870: %HasF.WithSelf.F.type.fd8 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %struct: %struct_type.x.c96 = struct_value (%int_2.ecc) [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.649: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.649 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.544: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.fb6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1a5, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.ef7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.544 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.fb6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.367: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.205: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.367, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.4ff: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Param.val: %Param = struct_value (%int_2.4ff) [concrete]
// CHECK:STDOUT:   %pattern_type.5a1: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.type.facet: %type = facet_value %C, () [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.type.665: type = fn_type @HasF.WithSelf.F, @HasF.WithSelf(%Param, %C.type.facet) [concrete]
// CHECK:STDOUT:   %HasF.WithSelf.F.625: %HasF.WithSelf.F.type.665 = struct_value () [concrete]
// CHECK:STDOUT:   %.6af: type = fn_type_with_self_type %HasF.WithSelf.F.type.fd8, %HasF.facet [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c85: %Int.as.Copy.impl.Op.type.b5d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.32d: <witness> = impl_witness imports.%Copy.impl_witness_table.07a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.4a0: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c4a: %Int.as.Copy.impl.Op.type.4a0 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.32d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.afe: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.a5f: type = fn_type_with_self_type %Copy.WithSelf.Op.type.afe, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.c4a, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.2: type = fn_type @Destroy.Op.loc22_27.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.2: %Destroy.Op.type.af7ec0.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.af7ec0.4: type = fn_type @Destroy.Op.loc22_27.4 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1dc86d.4: %Destroy.Op.type.af7ec0.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.649 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.dd3: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ec3)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1a5 = impl_witness_table (%Core.import_ref.dd3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.809: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c85)]
// CHECK:STDOUT:   %Copy.impl_witness_table.07a = impl_witness_table (%Core.import_ref.809), @Int.as.Copy.impl [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.a6d = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = splice_block %.loc4_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_17.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.67d)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.param_patt: %pattern_type.5a1 = value_param_pattern [concrete]
// CHECK:STDOUT:     %c.patt: %pattern_type.5a1 = at_binding_pattern c, %c.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = value_binding c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_17.2: type) {
// CHECK:STDOUT:   %T.loc4_17.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_17.1 (constants.%T.67d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_17.1)> [symbolic = %HasF.type (constants.%HasF.type.81d)]
// CHECK:STDOUT:   %Self.loc4_26.2: @HasF.%HasF.type (%HasF.type.81d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.fea)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.81d) = symbolic_binding Self, 1 [symbolic = %Self.loc4_26.2 (constants.%Self.fea)]
// CHECK:STDOUT:     %HasF.WithSelf.decl = interface_with_self_decl @HasF [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %HasF.WithSelf.F.decl: @HasF.WithSelf.%HasF.WithSelf.F.type (%HasF.WithSelf.F.type.cd0) = fn_decl @HasF.WithSelf.F [symbolic = @HasF.WithSelf.%HasF.WithSelf.F (constants.%HasF.WithSelf.F.252)] {
// CHECK:STDOUT:       %return.param_patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.51d1c4.1) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @HasF.WithSelf.F.%pattern_type (%pattern_type.51d1c4.1) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_17.2 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:       %.loc5_13.2: Core.Form = init_form %T.ref [symbolic = %.loc5_13.1 (constants.%.184347.1)]
// CHECK:STDOUT:       %return.param: ref @HasF.WithSelf.F.%T (%T.67d) = out_param call_param0
// CHECK:STDOUT:       %return: ref @HasF.WithSelf.F.%T (%T.67d) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.WithSelf.%HasF.assoc_type (%HasF.assoc_type.ea5) = assoc_entity element0, %HasF.WithSelf.F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.702)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_26.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @HasF.WithSelf.%assoc0.loc5_14.1
// CHECK:STDOUT:     .Param = <poisoned>
// CHECK:STDOUT:     witness = (@HasF.WithSelf.%HasF.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.8e1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.8e1 = return_slot_pattern %return.param_patt, %Param.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %Param.ref [concrete = constants.%.d1b]
// CHECK:STDOUT:     %return.param: ref %Param = out_param call_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   extend %HasF.type
// CHECK:STDOUT:   witness = %HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%struct_type.x.2bb [concrete = constants.%complete_type.fa5]
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
// CHECK:STDOUT:     %HasF.ref: %HasF.type.a6d = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [concrete = constants.%HasF.type.264]
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
// CHECK:STDOUT: generic fn @HasF.WithSelf.F(@HasF.%T.loc4_17.2: type, @HasF.%Self.loc4_26.1: @HasF.%HasF.type (%HasF.type.81d)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T.67d)]
// CHECK:STDOUT:   %.loc5_13.1: Core.Form = init_form %T [symbolic = %.loc5_13.1 (constants.%.184347.1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> out %return.param: @HasF.WithSelf.F.%T (%T.67d);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() -> out %return.param: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_21.1: %struct_type.x.c96 = struct_literal (%int_2) [concrete = constants.%struct]
// CHECK:STDOUT:   %impl.elem0: %.205 = impl_witness_access constants.%ImplicitAs.impl_witness.fb6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e6b]
// CHECK:STDOUT:   %bound_method.loc15_21.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_21.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc15_21.2(%int_2) [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = converted %int_2, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc15_21.3: ref %i32 = class_element_access %return.param, element0
// CHECK:STDOUT:   %.loc15_21.4: init %i32 to %.loc15_21.3 = in_place_init %.loc15_21.2 [concrete = constants.%int_2.4ff]
// CHECK:STDOUT:   %.loc15_21.5: init %Param to %return.param = class_init (%.loc15_21.4) [concrete = constants.%Param.val]
// CHECK:STDOUT:   %.loc15_22: init %Param = converted %.loc15_21.1, %.loc15_21.5 [concrete = constants.%Param.val]
// CHECK:STDOUT:   return %.loc15_22 to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c6b = value_binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc21_24: %HasF.assoc_type.2be = specific_constant @HasF.WithSelf.%assoc0.loc5_14.1, @HasF.WithSelf(constants.%Param, constants.%C.type.facet) [concrete = constants.%assoc0.b2a]
// CHECK:STDOUT:   %F.ref.loc21: %HasF.assoc_type.2be = name_ref F, %.loc21_24 [concrete = constants.%assoc0.b2a]
// CHECK:STDOUT:   %impl.elem0.loc21: %.6af = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc21_27.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc21: init %Param to %.loc21_27.1 = call %impl.elem0.loc21()
// CHECK:STDOUT:   %.loc21_27.2: ref %Param = temporary %.loc21_27.1, %C.as.HasF.impl.F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc21_28.1: ref %i32 = class_element_access %.loc21_27.2, element0
// CHECK:STDOUT:   %i32.loc21: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc21_28.2: %i32 = acquire_value %.loc21_28.1
// CHECK:STDOUT:   %a: %i32 = value_binding a, %.loc21_28.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.c6b = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.c6b = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_24: %HasF.assoc_type.2be = specific_constant @HasF.WithSelf.%assoc0.loc5_14.1, @HasF.WithSelf(constants.%Param, constants.%C.type.facet) [concrete = constants.%assoc0.b2a]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type.2be = name_ref F, %.loc22_24 [concrete = constants.%assoc0.b2a]
// CHECK:STDOUT:   %impl.elem0.loc22_24: %.6af = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc22_27.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc22: init %Param to %.loc22_27.1 = call %impl.elem0.loc22_24()
// CHECK:STDOUT:   %.loc22_27.2: ref %Param = temporary %.loc22_27.1, %C.as.HasF.impl.F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc22_28.1: ref %i32 = class_element_access %.loc22_27.2, element0
// CHECK:STDOUT:   %.loc22_28.2: %i32 = acquire_value %.loc22_28.1
// CHECK:STDOUT:   %impl.elem0.loc22_28: %.a5f = impl_witness_access constants.%Copy.impl_witness.32d, element0 [concrete = constants.%Int.as.Copy.impl.Op.c4a]
// CHECK:STDOUT:   %bound_method.loc22_28.1: <bound method> = bound_method %.loc22_28.2, %impl.elem0.loc22_28
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0.loc22_28, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc22_28.2: <bound method> = bound_method %.loc22_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc22_28.2(%.loc22_28.2)
// CHECK:STDOUT:   assign %b.var, %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   %i32.loc22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %b: ref %i32 = ref_binding b, %b.var
// CHECK:STDOUT:   %Destroy.Op.bound.loc22_27: <bound method> = bound_method %.loc22_27.2, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc22_27: init %empty_tuple.type = call %Destroy.Op.bound.loc22_27(%.loc22_27.2)
// CHECK:STDOUT:   %Destroy.Op.bound.loc22_3: <bound method> = bound_method %b.var, constants.%Destroy.Op.1dc86d.2
// CHECK:STDOUT:   %Destroy.Op.call.loc22_3: init %empty_tuple.type = call %Destroy.Op.bound.loc22_3(%b.var)
// CHECK:STDOUT:   %Destroy.Op.bound.loc21: <bound method> = bound_method %.loc21_27.2, constants.%Destroy.Op.1dc86d.4
// CHECK:STDOUT:   %Destroy.Op.call.loc21: init %empty_tuple.type = call %Destroy.Op.bound.loc21(%.loc21_27.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_27.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_27.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_27.3(%self.param: ref %struct_type.x.2bb) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc22_27.4(%self.param: ref %Param) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%T.67d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%T.67d, constants.%Self.fea) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%T.67d, constants.%Self.fea) {
// CHECK:STDOUT:   %T => constants.%T.67d
// CHECK:STDOUT:   %.loc5_13.1 => constants.%.184347.1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_17.1 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.264
// CHECK:STDOUT:   %Self.loc4_26.2 => constants.%Self.975
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Param, constants.%Self.fea) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.264
// CHECK:STDOUT:   %Self => constants.%Self.fea
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.7ea
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.1f9
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.2be
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.b2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.264
// CHECK:STDOUT:   %Self => constants.%HasF.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.fd8
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.870
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.2be
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.b2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf.F(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %.loc5_13.1 => constants.%.d1b
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.8e1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.WithSelf(constants.%Param, constants.%C.type.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.264
// CHECK:STDOUT:   %Self => constants.%C.type.facet
// CHECK:STDOUT:   %HasF.WithSelf.F.type => constants.%HasF.WithSelf.F.type.665
// CHECK:STDOUT:   %HasF.WithSelf.F => constants.%HasF.WithSelf.F.625
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.2be
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.b2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.1ec: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.1ec = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.010399.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f995b0.1: %I.type.010399.1 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.f995b0.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.882: type = pattern_type %Self.as_type [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.34b8b5.1: type = fn_type @I.WithSelf.F, @I.WithSelf(%T, %Self.f995b0.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.37d70d.1: %I.WithSelf.F.type.34b8b5.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.58ccb3.1: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8eaa79.1: %I.assoc_type.58ccb3.1 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.010399.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.as.I.impl.%I.impl_witness_table, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %Self.f995b0.2: %I.type.010399.2 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.34b8b5.2: type = fn_type @I.WithSelf.F, @I.WithSelf(%U, %Self.f995b0.1) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.37d70d.2: %I.WithSelf.F.type.34b8b5.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.58ccb3.2: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8eaa79.2: %I.assoc_type.58ccb3.2 = assoc_entity element0, @I.WithSelf.%I.WithSelf.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.f61: <witness> = require_complete_type %I.type.010399.2 [symbolic]
// CHECK:STDOUT:   %pattern_type.720: type = pattern_type %X [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.010399.2 = facet_value %X, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.type.932: type = fn_type @I.WithSelf.F, @I.WithSelf(%U, %I.facet) [symbolic]
// CHECK:STDOUT:   %I.WithSelf.F.29c: %I.WithSelf.F.type.932 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.783: <witness> = require_complete_type %X [symbolic]
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
// CHECK:STDOUT:   %I.decl: %I.type.1ec = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_13.1: type = splice_block %.loc8_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc8_10.2: type = symbolic_binding U, 0 [symbolic = %U.loc8_10.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_14.1)> [symbolic = %I.type (constants.%I.type.010399.1)]
// CHECK:STDOUT:   %Self.loc4_23.2: @I.%I.type (%I.type.010399.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.f995b0.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc4_23.1: @I.%I.type (%I.type.010399.1) = symbolic_binding Self, 1 [symbolic = %Self.loc4_23.2 (constants.%Self.f995b0.1)]
// CHECK:STDOUT:     %I.WithSelf.decl = interface_with_self_decl @I [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %I.WithSelf.F.decl: @I.WithSelf.%I.WithSelf.F.type (%I.WithSelf.F.type.34b8b5.1) = fn_decl @I.WithSelf.F [symbolic = @I.WithSelf.%I.WithSelf.F (constants.%I.WithSelf.F.37d70d.1)] {
// CHECK:STDOUT:       %self.param_patt: @I.WithSelf.F.%pattern_type.loc5_12 (%pattern_type.882) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @I.WithSelf.F.%pattern_type.loc5_12 (%pattern_type.882) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @I.WithSelf.F.%pattern_type.loc5_21 (%pattern_type.51d1c4.1) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @I.WithSelf.F.%pattern_type.loc5_21 (%pattern_type.51d1c4.1) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @I.WithSelf.F.%I.type (%I.type.010399.1) = specific_constant @I.%Self.loc4_23.1, @I(constants.%T) [symbolic = %Self (constants.%Self.f995b0.1)]
// CHECK:STDOUT:         %Self.ref: @I.WithSelf.F.%I.type (%I.type.010399.1) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.f995b0.1)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @I.WithSelf.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_14.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @I.WithSelf.F.%T (%T) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.WithSelf.%I.assoc_type (%I.assoc_type.58ccb3.1) = assoc_entity element0, %I.WithSelf.F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.8eaa79.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc4_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = @I.WithSelf.%assoc0.loc5_25.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     witness = (@I.WithSelf.%I.WithSelf.F.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @X.as.I.impl(@X.%U.loc8_10.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.1: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.010399.2)]
// CHECK:STDOUT:   %I.impl_witness.loc9_23.2: <witness> = impl_witness %I.impl_witness_table, @X.as.I.impl(%U) [symbolic = %I.impl_witness.loc9_23.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc9_21.1 [symbolic = %require_complete (constants.%require_complete.f61)]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic = %X.as.I.impl.F.type (constants.%X.as.I.impl.F.type)]
// CHECK:STDOUT:   %X.as.I.impl.F: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = struct_value () [symbolic = %X.as.I.impl.F (constants.%X.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.2 {
// CHECK:STDOUT:     %X.as.I.impl.F.decl: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = fn_decl @X.as.I.impl.F [symbolic = @X.as.I.impl.%X.as.I.impl.F (constants.%X.as.I.impl.F)] {
// CHECK:STDOUT:       %self.param_patt: @X.as.I.impl.F.%pattern_type.loc10_21 (%pattern_type.720) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @X.as.I.impl.F.%pattern_type.loc10_21 (%pattern_type.720) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %t.param_patt: @X.as.I.impl.F.%pattern_type.loc10_37 (%pattern_type.51d1c4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:       %t.patt: @X.as.I.impl.F.%pattern_type.loc10_37 (%pattern_type.51d1c4.2) = at_binding_pattern t, %t.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @X.as.I.impl.F.%X (%X) = value_param call_param0
// CHECK:STDOUT:       %.loc10_23.1: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc10_23.2: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_23.2 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @X.as.I.impl.F.%X (%X) = value_binding self, %self.param
// CHECK:STDOUT:       %t.param: @X.as.I.impl.F.%U (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_10.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %t: @X.as.I.impl.F.%U (%U) = value_binding t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (%X.as.I.impl.F.decl), @X.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness.loc9_23.1: <witness> = impl_witness %I.impl_witness_table, @X.as.I.impl(constants.%U) [symbolic = %I.impl_witness.loc9_23.2 (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %X.as.I.impl.F.decl
// CHECK:STDOUT:     extend %I.type.loc9_21.2
// CHECK:STDOUT:     witness = %I.impl_witness.loc9_23.1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_10.2: type) {
// CHECK:STDOUT:   %U.loc8_10.1: type = symbolic_binding U, 0 [symbolic = %U.loc8_10.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_10.1)> [symbolic = %I.type (constants.%I.type.010399.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.f61)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @X.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.1ec = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_10.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.2: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.010399.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc9: type = specific_constant @X.as.I.impl.%I.type.loc9_21.2, @X.as.I.impl(constants.%U) [symbolic = %I.type (constants.%I.type.010399.2)]
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
// CHECK:STDOUT: generic fn @I.WithSelf.F(@I.%T.loc4_14.2: type, @I.%Self.loc4_23.1: @I.%I.type (%I.type.010399.1)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.010399.1)]
// CHECK:STDOUT:   %Self: @I.WithSelf.F.%I.type (%I.type.010399.1) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f995b0.1)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:   %pattern_type.loc5_12: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_12 (constants.%pattern_type.882)]
// CHECK:STDOUT:   %pattern_type.loc5_21: type = pattern_type %T [symbolic = %pattern_type.loc5_21 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.WithSelf.F.%Self.as_type.loc5_14.1 (%Self.as_type), %t.param: @I.WithSelf.F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.as.I.impl.F(@X.%U.loc8_10.2: type) {
// CHECK:STDOUT:   %U: type = symbolic_binding U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type.loc10_21: type = pattern_type %X [symbolic = %pattern_type.loc10_21 (constants.%pattern_type.720)]
// CHECK:STDOUT:   %pattern_type.loc10_37: type = pattern_type %U [symbolic = %pattern_type.loc10_37 (constants.%pattern_type.51d1c4.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_21: <witness> = require_complete_type %X [symbolic = %require_complete.loc10_21 (constants.%require_complete.783)]
// CHECK:STDOUT:   %require_complete.loc10_37: <witness> = require_complete_type %U [symbolic = %require_complete.loc10_37 (constants.%require_complete.944)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @X.as.I.impl.F.%X (%X), %t.param: @X.as.I.impl.F.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%T, constants.%Self.f995b0.1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%T, constants.%Self.f995b0.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.1
// CHECK:STDOUT:   %Self => constants.%Self.f995b0.1
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT:   %pattern_type.loc5_12 => constants.%pattern_type.882
// CHECK:STDOUT:   %pattern_type.loc5_21 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_10.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.2
// CHECK:STDOUT:   %Self.loc4_23.2 => constants.%Self.f995b0.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.1 => constants.%I.type.010399.2
// CHECK:STDOUT:   %I.impl_witness.loc9_23.2 => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.f61
// CHECK:STDOUT:   %X.as.I.impl.F.type => constants.%X.as.I.impl.F.type
// CHECK:STDOUT:   %X.as.I.impl.F => constants.%X.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%U, constants.%Self.f995b0.1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.2
// CHECK:STDOUT:   %Self => constants.%Self.f995b0.1
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.34b8b5.2
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.37d70d.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.58ccb3.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.8eaa79.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl.F(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type.loc10_21 => constants.%pattern_type.720
// CHECK:STDOUT:   %pattern_type.loc10_37 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf(constants.%U, constants.%I.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %I.WithSelf.F.type => constants.%I.WithSelf.F.type.932
// CHECK:STDOUT:   %I.WithSelf.F => constants.%I.WithSelf.F.29c
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.58ccb3.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.8eaa79.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.WithSelf.F(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.010399.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%X
// CHECK:STDOUT:   %pattern_type.loc5_12 => constants.%pattern_type.720
// CHECK:STDOUT:   %pattern_type.loc5_21 => constants.%pattern_type.51d1c4.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
