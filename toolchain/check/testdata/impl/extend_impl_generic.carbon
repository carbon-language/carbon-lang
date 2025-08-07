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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %HasF.type.fe3: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.901: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.322: %HasF.type.901 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %HasF.F.type.46c: type = fn_type @HasF.F, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %HasF.F.823: %HasF.F.type.46c = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.dd8: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.d31: %HasF.assoc_type.dd8 = assoc_entity element0, @HasF.%HasF.F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.2b9: <witness> = impl_witness @Param.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.756: type = ptr_type %Param [concrete]
// CHECK:STDOUT:   %pattern_type.fae: type = pattern_type %ptr.756 [concrete]
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.type: type = fn_type @Param.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op: %Param.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type.b18: type = facet_type <@HasF, @HasF(%Param)> [concrete]
// CHECK:STDOUT:   %Self.a41: %HasF.type.b18 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %HasF.F.type.7f1: type = fn_type @HasF.F, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %HasF.F.eff: %HasF.F.type.7f1 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.257: type = assoc_entity_type @HasF, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %assoc0.afb: %HasF.assoc_type.257 = assoc_entity element0, @HasF.%HasF.F.decl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness @C.%HasF.impl_witness_table [concrete]
// CHECK:STDOUT:   %pattern_type.b01: type = pattern_type %Param [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F.type: type = fn_type @C.as.HasF.impl.F [concrete]
// CHECK:STDOUT:   %C.as.HasF.impl.F: %C.as.HasF.impl.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.b18 = facet_value %C, (%HasF.impl_witness) [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness.edd: <witness> = impl_witness @C.%Destroy.impl_witness_table [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.type: type = fn_type @C.as.Destroy.impl.Op [concrete]
// CHECK:STDOUT:   %C.as.Destroy.impl.Op: %C.as.Destroy.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f06: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.956: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.956 [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.956, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Param.val: %Param = struct_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.22c: type = fn_type_with_self_type %HasF.F.type.7f1, %HasF.facet [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.type.a17: type = fn_type @T.as.Destroy.impl.Op, @T.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.e6a: %T.as.Destroy.impl.Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %T.as.Destroy.impl.Op.e6a, @T.as.Destroy.impl.Op(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:   %HasF.decl: %HasF.type.fe3 = interface_decl @HasF [concrete = constants.%HasF.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:     %c.param_patt: %pattern_type.c48 = value_param_pattern %c.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.2: type) {
// CHECK:STDOUT:   %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.1)> [symbolic = %HasF.type (constants.%HasF.type.901)]
// CHECK:STDOUT:   %Self.2: @HasF.%HasF.type (%HasF.type.901) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:   %HasF.F.type: type = fn_type @HasF.F, @HasF(%T.loc4_16.1) [symbolic = %HasF.F.type (constants.%HasF.F.type.46c)]
// CHECK:STDOUT:   %HasF.F: @HasF.%HasF.F.type (%HasF.F.type.46c) = struct_value () [symbolic = %HasF.F (constants.%HasF.F.823)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF, @HasF(%T.loc4_16.1) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.dd8)]
// CHECK:STDOUT:   %assoc0.loc5_14.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.dd8) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.901) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:     %HasF.F.decl: @HasF.%HasF.F.type (%HasF.F.type.46c) = fn_decl @HasF.F [symbolic = @HasF.%HasF.F (constants.%HasF.F.823)] {
// CHECK:STDOUT:       %return.patt: @HasF.F.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @HasF.F.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @HasF.F.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @HasF.F.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.dd8) = assoc_entity element0, %HasF.F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_14.1
// CHECK:STDOUT:     .Param = <poisoned>
// CHECK:STDOUT:     witness = (%HasF.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Param.as.Destroy.impl: constants.%Param as constants.%Destroy.type {
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.decl: %Param.as.Destroy.impl.Op.type = fn_decl @Param.as.Destroy.impl.Op [concrete = constants.%Param.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.fae = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.fae = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc8: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.756 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%Param [concrete = constants.%Param]
// CHECK:STDOUT:     %self: %ptr.756 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %Param.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @Param.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.HasF.impl: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %C.as.HasF.impl.F.decl: %C.as.HasF.impl.F.type = fn_decl @C.as.HasF.impl.F [concrete = constants.%C.as.HasF.impl.F] {
// CHECK:STDOUT:     %return.patt: %pattern_type.b01 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.b01 = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param call_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = %C.as.HasF.impl.F.decl
// CHECK:STDOUT:   witness = @C.%HasF.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Destroy.impl: constants.%C as constants.%Destroy.type {
// CHECK:STDOUT:   %C.as.Destroy.impl.Op.decl: %C.as.Destroy.impl.Op.type = fn_decl @C.as.Destroy.impl.Op [concrete = constants.%C.as.Destroy.impl.Op] {
// CHECK:STDOUT:     %self.patt: %pattern_type.44a = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.44a = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %.loc12: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %ptr.019 = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %ptr.019 = bind_name self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %C.as.Destroy.impl.Op.decl
// CHECK:STDOUT:   witness = @C.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc9: %Param.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   impl_decl @Param.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@Param.as.Destroy.impl.%Param.as.Destroy.impl.Op.decl), @Param.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.2b9]
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
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [concrete = constants.%HasF.type.b18]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HasF.impl_witness_table = impl_witness_table (@C.as.HasF.impl.%C.as.HasF.impl.F.decl), @C.as.HasF.impl [concrete]
// CHECK:STDOUT:   %HasF.impl_witness: <witness> = impl_witness %HasF.impl_witness_table [concrete = constants.%HasF.impl_witness]
// CHECK:STDOUT:   impl_decl @C.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (@C.as.Destroy.impl.%C.as.Destroy.impl.Op.decl), @C.as.Destroy.impl [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table [concrete = constants.%Destroy.impl_witness.edd]
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
// CHECK:STDOUT: generic fn @HasF.F(@HasF.%T.loc4_16.2: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.901)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @HasF.F.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Param.as.Destroy.impl.Op(%self.param: %ptr.756) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @C.as.HasF.impl.F() -> %return.param: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_21.1: %struct_type.x.c96 = struct_literal (%int_2)
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.956]
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
// CHECK:STDOUT: fn @C.as.Destroy.impl.Op(%self.param: %ptr.019) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.7ce = binding_pattern a [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc21_17: %HasF.assoc_type.257 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %F.ref.loc21: %HasF.assoc_type.257 = name_ref F, %.loc21_17 [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %impl.elem0.loc21: %.22c = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc21_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc21: init %Param = call %impl.elem0.loc21() to %.loc21_20.1
// CHECK:STDOUT:   %.loc21_20.2: ref %Param = temporary %.loc21_20.1, %C.as.HasF.impl.F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc21_21.1: ref %i32 = class_element_access %.loc21_20.2, element0
// CHECK:STDOUT:   %.loc21_10: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21_21.2: %i32 = bind_value %.loc21_21.1
// CHECK:STDOUT:   %a: %i32 = bind_name a, %.loc21_21.2
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.7ce = binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.7ce = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var %b.var_patt
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_17: %HasF.assoc_type.257 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type.257 = name_ref F, %.loc22_17 [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %impl.elem0.loc22: %.22c = impl_witness_access constants.%HasF.impl_witness, element0 [concrete = constants.%C.as.HasF.impl.F]
// CHECK:STDOUT:   %.loc22_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %C.as.HasF.impl.F.call.loc22: init %Param = call %impl.elem0.loc22() to %.loc22_20.1
// CHECK:STDOUT:   %.loc22_20.2: ref %Param = temporary %.loc22_20.1, %C.as.HasF.impl.F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %Param.elem = name_ref x, @Param.%.loc9 [concrete = @Param.%.loc9]
// CHECK:STDOUT:   %.loc22_21.1: ref %i32 = class_element_access %.loc22_20.2, element0
// CHECK:STDOUT:   %.loc22_21.2: %i32 = bind_value %.loc22_21.1
// CHECK:STDOUT:   assign %b.var, %.loc22_21.2
// CHECK:STDOUT:   %.loc22_10: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.bound.loc22: <bound method> = bound_method %.loc22_20.1, constants.%Param.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc22_20: %ptr.756 = addr_of %.loc22_20.1
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.call.loc22: init %empty_tuple.type = call %Param.as.Destroy.impl.Op.bound.loc22(%addr.loc22_20)
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.bound: <bound method> = bound_method %b.var, constants.%T.as.Destroy.impl.Op.e6a
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%T.as.Destroy.impl.Op.e6a, @T.as.Destroy.impl.Op(constants.%i32) [concrete = constants.%T.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %b.var, %T.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr.loc22_3: %ptr.235 = addr_of %b.var
// CHECK:STDOUT:   %T.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method(%addr.loc22_3)
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.bound.loc21: <bound method> = bound_method %.loc21_20.1, constants.%Param.as.Destroy.impl.Op
// CHECK:STDOUT:   %addr.loc21: %ptr.756 = addr_of %.loc21_20.1
// CHECK:STDOUT:   %Param.as.Destroy.impl.Op.call.loc21: init %empty_tuple.type = call %Param.as.Destroy.impl.Op.bound.loc21(%addr.loc21)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%T, constants.%Self.322) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.1 => constants.%Param
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b18
// CHECK:STDOUT:   %Self.2 => constants.%Self.a41
// CHECK:STDOUT:   %HasF.F.type => constants.%HasF.F.type.7f1
// CHECK:STDOUT:   %HasF.F => constants.%HasF.F.eff
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.257
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.afb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF.F(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b01
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325e65.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209206.1: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.928: type = facet_access_type %Self.209206.1 [symbolic]
// CHECK:STDOUT:   %pattern_type.4be: type = pattern_type %Self.as_type.928 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %I.F.type.2aef59.1: type = fn_type @I.F, @I(%T) [symbolic]
// CHECK:STDOUT:   %I.F.bb2dd4.1: %I.F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.1: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.1: %I.assoc_type.1e5078.1 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.325e65.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self.209206.2: %I.type.325e65.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %I.F.type.2aef59.2: type = fn_type @I.F, @I(%U) [symbolic]
// CHECK:STDOUT:   %I.F.bb2dd4.2: %I.F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.2: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.2: %I.assoc_type.1e5078.2 = assoc_entity element0, @I.%I.F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.%I.impl_witness_table, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %pattern_type.830: type = pattern_type %X [symbolic]
// CHECK:STDOUT:   %pattern_type.7dcd0a.2: type = pattern_type %U [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic]
// CHECK:STDOUT:   %X.as.I.impl.F: %X.as.I.impl.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.325e65.2 = facet_value %X, (%I.impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.f6d: type = pattern_type auto [concrete]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @X.%Destroy.impl_witness_table, @X.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %ptr.428: type = ptr_type %X [symbolic]
// CHECK:STDOUT:   %pattern_type.d72: type = pattern_type %ptr.428 [symbolic]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op, @X.as.Destroy.impl(%U) [symbolic]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: %X.as.Destroy.impl.Op.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc8_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.1)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:   %I.F.type: type = fn_type @I.F, @I(%T.loc4_13.1) [symbolic = %I.F.type (constants.%I.F.type.2aef59.1)]
// CHECK:STDOUT:   %I.F: @I.%I.F.type (%I.F.type.2aef59.1) = struct_value () [symbolic = %I.F (constants.%I.F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.1) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5078.1)]
// CHECK:STDOUT:   %assoc0.loc5_25.2: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:     %I.F.decl: @I.%I.F.type (%I.F.type.2aef59.1) = fn_decl @I.F [symbolic = @I.%I.F (constants.%I.F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @I.F.%pattern_type.loc5_8 (%pattern_type.4be) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @I.F.%pattern_type.loc5_8 (%pattern_type.4be) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @I.F.%pattern_type.loc5_20 (%pattern_type.7dcd0a.1) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @I.F.%pattern_type.loc5_20 (%pattern_type.7dcd0a.1) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @I.F.%Self.as_type.loc5_14.1 (%Self.as_type.928) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.928)] {
// CHECK:STDOUT:         %.loc5_14.2: @I.F.%I.type (%I.type.325e65.1) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.ref: @I.F.%I.type (%I.type.325e65.1) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @I.F.%Self.as_type.loc5_14.1 (%Self.as_type.928) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @I.F.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @I.F.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %I.F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_25.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     witness = (%I.F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @X.as.I.impl(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.1: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type.loc9_21.1 [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:   %I.impl_witness: <witness> = impl_witness @X.%I.impl_witness_table, @X.as.I.impl(%U) [symbolic = %I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.as.I.impl.F.type: type = fn_type @X.as.I.impl.F, @X.as.I.impl(%U) [symbolic = %X.as.I.impl.F.type (constants.%X.as.I.impl.F.type)]
// CHECK:STDOUT:   %X.as.I.impl.F: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = struct_value () [symbolic = %X.as.I.impl.F (constants.%X.as.I.impl.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.2 {
// CHECK:STDOUT:     %X.as.I.impl.F.decl: @X.as.I.impl.%X.as.I.impl.F.type (%X.as.I.impl.F.type) = fn_decl @X.as.I.impl.F [symbolic = @X.as.I.impl.%X.as.I.impl.F (constants.%X.as.I.impl.F)] {
// CHECK:STDOUT:       %self.patt: @X.as.I.impl.F.%pattern_type.loc10_10 (%pattern_type.830) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @X.as.I.impl.F.%pattern_type.loc10_10 (%pattern_type.830) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %t.patt: @X.as.I.impl.F.%pattern_type.loc10_22 (%pattern_type.7dcd0a.2) = binding_pattern t [concrete]
// CHECK:STDOUT:       %t.param_patt: @X.as.I.impl.F.%pattern_type.loc10_22 (%pattern_type.7dcd0a.2) = value_param_pattern %t.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @X.as.I.impl.F.%X (%X) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc10_16.2: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_16.2 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @X.as.I.impl.F.%X (%X) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @X.as.I.impl.F.%U (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %t: @X.as.I.impl.F.%U (%U) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %X.as.I.impl.F.decl
// CHECK:STDOUT:     witness = @X.%I.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @X.as.Destroy.impl(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @X.%Destroy.impl_witness_table, @X.as.Destroy.impl(%U) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.as.Destroy.impl.Op.type: type = fn_type @X.as.Destroy.impl.Op, @X.as.Destroy.impl(%U) [symbolic = %X.as.Destroy.impl.Op.type (constants.%X.as.Destroy.impl.Op.type)]
// CHECK:STDOUT:   %X.as.Destroy.impl.Op: @X.as.Destroy.impl.%X.as.Destroy.impl.Op.type (%X.as.Destroy.impl.Op.type) = struct_value () [symbolic = %X.as.Destroy.impl.Op (constants.%X.as.Destroy.impl.Op)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: constants.%X as constants.%Destroy.type {
// CHECK:STDOUT:     %X.as.Destroy.impl.Op.decl: @X.as.Destroy.impl.%X.as.Destroy.impl.Op.type (%X.as.Destroy.impl.Op.type) = fn_decl @X.as.Destroy.impl.Op [symbolic = @X.as.Destroy.impl.%X.as.Destroy.impl.Op (constants.%X.as.Destroy.impl.Op)] {
// CHECK:STDOUT:       %self.patt: @X.as.Destroy.impl.Op.%pattern_type (%pattern_type.d72) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @X.as.Destroy.impl.Op.%pattern_type (%pattern_type.d72) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %.loc8_19.1: %pattern_type.f6d = addr_pattern %self.param_patt [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @X.as.Destroy.impl.Op.%ptr (%ptr.428) = value_param call_param0
// CHECK:STDOUT:       %.loc8_19.2: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc8_19.3: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc8_19.3 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @X.as.Destroy.impl.Op.%ptr (%ptr.428) = bind_name self, %self.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Op = %X.as.Destroy.impl.Op.decl
// CHECK:STDOUT:     witness = @X.%Destroy.impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U.loc8_9.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.1)> [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.type [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @X.as.I.impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.2 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.2: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.1 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %I.impl_witness_table = impl_witness_table (@X.as.I.impl.%X.as.I.impl.F.decl), @X.as.I.impl [concrete]
// CHECK:STDOUT:     %I.impl_witness: <witness> = impl_witness %I.impl_witness_table, @X.as.I.impl(constants.%U) [symbolic = @X.as.I.impl.%I.impl_witness (constants.%I.impl_witness)]
// CHECK:STDOUT:     %.loc9: type = specific_constant @X.as.I.impl.%I.type.loc9_21.2, @X.as.I.impl(constants.%U) [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     impl_decl @X.as.Destroy.impl [concrete] {} {}
// CHECK:STDOUT:     %Destroy.impl_witness_table = impl_witness_table (@X.as.Destroy.impl.%X.as.Destroy.impl.Op.decl), @X.as.Destroy.impl [concrete]
// CHECK:STDOUT:     %Destroy.impl_witness: <witness> = impl_witness %Destroy.impl_witness_table, @X.as.Destroy.impl(constants.%U) [symbolic = @X.as.Destroy.impl.%Destroy.impl_witness (constants.%Destroy.impl_witness)]
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
// CHECK:STDOUT: generic fn @I.F(@I.%T.loc4_13.2: type, @I.%Self.1: @I.%I.type (%I.type.325e65.1)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self: @I.F.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type.928)]
// CHECK:STDOUT:   %pattern_type.loc5_8: type = pattern_type %Self.as_type.loc5_14.1 [symbolic = %pattern_type.loc5_8 (constants.%pattern_type.4be)]
// CHECK:STDOUT:   %pattern_type.loc5_20: type = pattern_type %T [symbolic = %pattern_type.loc5_20 (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @I.F.%Self.as_type.loc5_14.1 (%Self.as_type.928), %t.param: @I.F.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.as.I.impl.F(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %pattern_type.loc10_10: type = pattern_type %X [symbolic = %pattern_type.loc10_10 (constants.%pattern_type.830)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete.loc10_25: <witness> = require_complete_type %I.type [symbolic = %require_complete.loc10_25 (constants.%require_complete.cfe)]
// CHECK:STDOUT:   %pattern_type.loc10_22: type = pattern_type %U [symbolic = %pattern_type.loc10_22 (constants.%pattern_type.7dcd0a.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_14: <witness> = require_complete_type %X [symbolic = %require_complete.loc10_14 (constants.%require_complete.441)]
// CHECK:STDOUT:   %require_complete.loc10_23: <witness> = require_complete_type %U [symbolic = %require_complete.loc10_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @X.as.I.impl.F.%X (%X), %t.param: @X.as.I.impl.F.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @X.as.Destroy.impl.Op(@X.%U.loc8_9.2: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %ptr: type = ptr_type %X [symbolic = %ptr (constants.%ptr.428)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr [symbolic = %pattern_type (constants.%pattern_type.d72)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @X.as.Destroy.impl.Op.%ptr (%ptr.428)) = "no_op";
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%T, constants.%Self.209206.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.1
// CHECK:STDOUT:   %Self => constants.%Self.209206.1
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type.928
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.4be
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.209206.2
// CHECK:STDOUT:   %I.F.type => constants.%I.F.type.2aef59.2
// CHECK:STDOUT:   %I.F => constants.%I.F.bb2dd4.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e5078.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.8f0422.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.1 => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT:   %I.impl_witness => constants.%I.impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %X.as.I.impl.F.type => constants.%X.as.I.impl.F.type
// CHECK:STDOUT:   %X.as.I.impl.F => constants.%X.as.I.impl.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.I.impl.F(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %pattern_type.loc10_10 => constants.%pattern_type.830
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete.loc10_25 => constants.%require_complete.cfe
// CHECK:STDOUT:   %pattern_type.loc10_22 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I.F(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%X
// CHECK:STDOUT:   %pattern_type.loc5_8 => constants.%pattern_type.830
// CHECK:STDOUT:   %pattern_type.loc5_20 => constants.%pattern_type.7dcd0a.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.Destroy.impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %Destroy.impl_witness => constants.%Destroy.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X.as.Destroy.impl.Op(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %ptr => constants.%ptr.428
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.d72
// CHECK:STDOUT: }
// CHECK:STDOUT:
