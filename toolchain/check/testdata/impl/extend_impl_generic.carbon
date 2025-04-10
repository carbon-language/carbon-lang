// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %HasF.type.fe3: type = generic_interface_type @HasF [concrete]
// CHECK:STDOUT:   %HasF.generic: %HasF.type.fe3 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.type.901: type = facet_type <@HasF, @HasF(%T)> [symbolic]
// CHECK:STDOUT:   %Self.322: %HasF.type.901 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.46c: type = fn_type @F.1, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %F.823: %F.type.46c = struct_value () [symbolic]
// CHECK:STDOUT:   %HasF.assoc_type.dd8: type = assoc_entity_type @HasF, @HasF(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.d31: %HasF.assoc_type.dd8 = assoc_entity element0, @HasF.%F.decl [symbolic]
// CHECK:STDOUT:   %Param: type = class_type @Param [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %Param.elem: type = unbound_element_type %Param, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.x.ed6: type = struct_type {.x: %i32} [concrete]
// CHECK:STDOUT:   %complete_type.1ec: <witness> = complete_type_witness %struct_type.x.ed6 [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %HasF.type.b18: type = facet_type <@HasF, @HasF(%Param)> [concrete]
// CHECK:STDOUT:   %Self.a41: %HasF.type.b18 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.7f1: type = fn_type @F.1, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %F.eff: %F.type.7f1 = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.assoc_type.257: type = assoc_entity_type @HasF, @HasF(%Param) [concrete]
// CHECK:STDOUT:   %assoc0.afb: %HasF.assoc_type.257 = assoc_entity element0, @HasF.%F.decl [concrete]
// CHECK:STDOUT:   %impl_witness.9bf: <witness> = impl_witness (@impl.3b1.%F.decl) [concrete]
// CHECK:STDOUT:   %F.type.94c: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.901: %F.type.94c = struct_value () [concrete]
// CHECK:STDOUT:   %HasF.facet: %HasF.type.b18 = facet_value %C, (%impl_witness.9bf) [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %struct_type.x.c96: type = struct_type {.x: Core.IntLiteral} [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%impl_witness.d39) [concrete]
// CHECK:STDOUT:   %.be7: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_2.ecc, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_2.ecc, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_2.ef8: %i32 = int_value 2 [concrete]
// CHECK:STDOUT:   %Param.val: %Param = struct_value (%int_2.ef8) [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %.75e: type = fn_type_with_self_type %F.type.7f1, %HasF.facet [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %T.patt.loc4_16.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_16.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Param.decl: type = class_decl @Param [concrete = constants.%Param] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %c.patt: %C = binding_pattern c
// CHECK:STDOUT:     %c.param_patt: %C = value_param_pattern %c.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %c.param: %C = value_param call_param0
// CHECK:STDOUT:     %C.ref.loc20: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: %C = bind_name c, %c.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @HasF(%T.loc4_16.1: type) {
// CHECK:STDOUT:   %T.loc4_16.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_16.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_16.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_16.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type: type = facet_type <@HasF, @HasF(%T.loc4_16.2)> [symbolic = %HasF.type (constants.%HasF.type.901)]
// CHECK:STDOUT:   %Self.2: @HasF.%HasF.type (%HasF.type.901) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @HasF(%T.loc4_16.2) [symbolic = %F.type (constants.%F.type.46c)]
// CHECK:STDOUT:   %F: @HasF.%F.type (%F.type.46c) = struct_value () [symbolic = %F (constants.%F.823)]
// CHECK:STDOUT:   %HasF.assoc_type: type = assoc_entity_type @HasF, @HasF(%T.loc4_16.2) [symbolic = %HasF.assoc_type (constants.%HasF.assoc_type.dd8)]
// CHECK:STDOUT:   %assoc0.loc5_14.2: @HasF.%HasF.assoc_type (%HasF.assoc_type.dd8) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @HasF.%HasF.type (%HasF.type.901) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.322)]
// CHECK:STDOUT:     %F.decl: @HasF.%F.type (%F.type.46c) = fn_decl @F.1 [symbolic = @HasF.%F (constants.%F.823)] {
// CHECK:STDOUT:       %return.patt: @F.1.%T (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.1.%T (%T) = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @HasF.%T.loc4_16.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %return.param: ref @F.1.%T (%T) = out_param call_param0
// CHECK:STDOUT:       %return: ref @F.1.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_14.1: @HasF.%HasF.assoc_type (%HasF.assoc_type.dd8) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_14.2 (constants.%assoc0.d31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_14.1
// CHECK:STDOUT:     .Param = <poisoned>
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.3b1: %Self.ref as %HasF.type {
// CHECK:STDOUT:   %F.decl: %F.type.94c = fn_decl @F.2 [concrete = constants.%F.901] {
// CHECK:STDOUT:     %return.patt: %Param = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %Param = out_param_pattern %return.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %return.param: ref %Param = out_param call_param0
// CHECK:STDOUT:     %return: ref %Param = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT:   witness = @C.%impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Param {
// CHECK:STDOUT:   %.loc9_8: %Param.elem = field_decl x, element0 [concrete]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %.loc9_3: %Param.elem = var_pattern %.loc9_8
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.var: ref %Param.elem = var <none>
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.x.ed6 [concrete = constants.%complete_type.1ec]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Param
// CHECK:STDOUT:   .x = %.loc9_8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   impl_decl @impl.3b1 [concrete] {} {
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %HasF.ref: %HasF.type.fe3 = name_ref HasF, file.%HasF.decl [concrete = constants.%HasF.generic]
// CHECK:STDOUT:     %Param.ref: type = name_ref Param, file.%Param.decl [concrete = constants.%Param]
// CHECK:STDOUT:     %HasF.type: type = facet_type <@HasF, @HasF(constants.%Param)> [concrete = constants.%HasF.type.b18]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.3b1.%F.decl) [concrete = constants.%impl_witness.9bf]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .HasF = <poisoned>
// CHECK:STDOUT:   .Param = <poisoned>
// CHECK:STDOUT:   .F = <poisoned>
// CHECK:STDOUT:   extend @impl.3b1.%HasF.type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.1(@HasF.%T.loc4_16.1: type, @HasF.%Self.1: @HasF.%HasF.type (%HasF.type.901)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() -> @F.1.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.2() -> %return.param_patt: %Param {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc15_21.1: %struct_type.x.c96 = struct_literal (%int_2)
// CHECK:STDOUT:   %impl.elem0: %.be7 = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc15_21.1: <bound method> = bound_method %int_2, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_21.2: <bound method> = bound_method %int_2, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc15_21.2(%int_2) [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.2: init %i32 = converted %int_2, %int.convert_checked [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.3: ref %i32 = class_element_access %return, element0
// CHECK:STDOUT:   %.loc15_21.4: init %i32 = initialize_from %.loc15_21.2 to %.loc15_21.3 [concrete = constants.%int_2.ef8]
// CHECK:STDOUT:   %.loc15_21.5: init %Param = class_init (%.loc15_21.4), %return [concrete = constants.%Param.val]
// CHECK:STDOUT:   %.loc15_22: init %Param = converted %.loc15_21.1, %.loc15_21.5 [concrete = constants.%Param.val]
// CHECK:STDOUT:   return %.loc15_22 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%c.param_patt: %C) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %i32 = binding_pattern a
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.ref.loc21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %.loc21_17: %HasF.assoc_type.257 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %F.ref.loc21: %HasF.assoc_type.257 = name_ref F, %.loc21_17 [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %impl.elem0.loc21: %.75e = impl_witness_access constants.%impl_witness.9bf, element0 [concrete = constants.%F.901]
// CHECK:STDOUT:   %.loc21_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc21: init %Param = call %impl.elem0.loc21() to %.loc21_20.1
// CHECK:STDOUT:   %.loc21_20.2: ref %Param = temporary %.loc21_20.1, %F.call.loc21
// CHECK:STDOUT:   %x.ref.loc21: %Param.elem = name_ref x, @Param.%.loc9_8 [concrete = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc21_21: ref %i32 = class_element_access %.loc21_20.2, element0
// CHECK:STDOUT:   %.loc21_10: type = splice_block %i32.loc21 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc21: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc21: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %i32 = bind_name a, %.loc21_21
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %i32 = binding_pattern b
// CHECK:STDOUT:     %.loc22_3: %i32 = var_pattern %b.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %i32 = var b
// CHECK:STDOUT:   %c.ref: %C = name_ref c, %c
// CHECK:STDOUT:   %.loc22_17: %HasF.assoc_type.257 = specific_constant @HasF.%assoc0.loc5_14.1, @HasF(constants.%Param) [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %F.ref.loc22: %HasF.assoc_type.257 = name_ref F, %.loc22_17 [concrete = constants.%assoc0.afb]
// CHECK:STDOUT:   %impl.elem0.loc22: %.75e = impl_witness_access constants.%impl_witness.9bf, element0 [concrete = constants.%F.901]
// CHECK:STDOUT:   %.loc22_20.1: ref %Param = temporary_storage
// CHECK:STDOUT:   %F.call.loc22: init %Param = call %impl.elem0.loc22() to %.loc22_20.1
// CHECK:STDOUT:   %.loc22_20.2: ref %Param = temporary %.loc22_20.1, %F.call.loc22
// CHECK:STDOUT:   %x.ref.loc22: %Param.elem = name_ref x, @Param.%.loc9_8 [concrete = @Param.%.loc9_8]
// CHECK:STDOUT:   %.loc22_21.1: ref %i32 = class_element_access %.loc22_20.2, element0
// CHECK:STDOUT:   %.loc22_21.2: %i32 = bind_value %.loc22_21.1
// CHECK:STDOUT:   assign %b.var, %.loc22_21.2
// CHECK:STDOUT:   %.loc22_10: type = splice_block %i32.loc22 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %i32 = bind_name b, %b.var
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%T) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.322) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(%T.loc4_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HasF(constants.%Param) {
// CHECK:STDOUT:   %T.loc4_16.2 => constants.%Param
// CHECK:STDOUT:   %T.patt.loc4_16.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %HasF.type => constants.%HasF.type.b18
// CHECK:STDOUT:   %Self.2 => constants.%Self.a41
// CHECK:STDOUT:   %F.type => constants.%F.type.7f1
// CHECK:STDOUT:   %F => constants.%F.eff
// CHECK:STDOUT:   %HasF.assoc_type => constants.%HasF.assoc_type.257
// CHECK:STDOUT:   %assoc0.loc5_14.2 => constants.%assoc0.afb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%Param, constants.%HasF.facet) {
// CHECK:STDOUT:   %T => constants.%Param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- extend_impl_generic_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type.dac: type = generic_interface_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type.dac = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.325e65.1: type = facet_type <@I, @I(%T)> [symbolic]
// CHECK:STDOUT:   %Self.209206.1: %I.type.325e65.1 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self.209206.1 [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.1: type = fn_type @F.1, @I(%T) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.1: %F.type.2aef59.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.1: type = assoc_entity_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.1: %I.assoc_type.1e5078.1 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 0 [symbolic]
// CHECK:STDOUT:   %X.type: type = generic_class_type @X [concrete]
// CHECK:STDOUT:   %X.generic: %X.type = struct_value () [concrete]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic]
// CHECK:STDOUT:   %I.type.325e65.2: type = facet_type <@I, @I(%U)> [symbolic]
// CHECK:STDOUT:   %Self.209206.2: %I.type.325e65.2 = bind_symbolic_name Self, 1 [symbolic]
// CHECK:STDOUT:   %F.type.2aef59.2: type = fn_type @F.1, @I(%U) [symbolic]
// CHECK:STDOUT:   %F.bb2dd4.2: %F.type.2aef59.2 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.assoc_type.1e5078.2: type = assoc_entity_type @I, @I(%U) [symbolic]
// CHECK:STDOUT:   %assoc0.8f0422.2: %I.assoc_type.1e5078.2 = assoc_entity element0, @I.%F.decl [symbolic]
// CHECK:STDOUT:   %require_complete.cfe: <witness> = require_complete_type %I.type.325e65.2 [symbolic]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.type.e88: type = fn_type @F.2, @impl(%U) [symbolic]
// CHECK:STDOUT:   %F.b02: %F.type.e88 = struct_value () [symbolic]
// CHECK:STDOUT:   %I.facet: %I.type.325e65.2 = facet_value %X, (%impl_witness) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.441: <witness> = require_complete_type %X [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %U [symbolic]
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
// CHECK:STDOUT:   %I.decl: %I.type.dac = interface_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %X.decl: %X.type = class_decl @X [concrete = constants.%X.generic] {
// CHECK:STDOUT:     %U.patt.loc8_9.1: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %U.loc8_9.1: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @I(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T.loc4_13.2)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self.2: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:   %F.type: type = fn_type @F.1, @I(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type.2aef59.1)]
// CHECK:STDOUT:   %F: @I.%F.type (%F.type.2aef59.1) = struct_value () [symbolic = %F (constants.%F.bb2dd4.1)]
// CHECK:STDOUT:   %I.assoc_type: type = assoc_entity_type @I, @I(%T.loc4_13.2) [symbolic = %I.assoc_type (constants.%I.assoc_type.1e5078.1)]
// CHECK:STDOUT:   %assoc0.loc5_25.2: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.1: @I.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self.2 (constants.%Self.209206.1)]
// CHECK:STDOUT:     %F.decl: @I.%F.type (%F.type.2aef59.1) = fn_decl @F.1 [symbolic = @I.%F (constants.%F.bb2dd4.1)] {
// CHECK:STDOUT:       %self.patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %t.patt: @F.1.%T (%T) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.1.%T (%T) = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = value_param call_param0
// CHECK:STDOUT:       %.loc5_14.1: type = splice_block %.loc5_14.3 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)] {
// CHECK:STDOUT:         %.loc5_14.2: @F.1.%I.type (%I.type.325e65.1) = specific_constant @I.%Self.1, @I(constants.%T) [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.ref: @F.1.%I.type (%I.type.325e65.1) = name_ref Self, %.loc5_14.2 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:         %Self.as_type.loc5_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:         %.loc5_14.3: type = converted %Self.ref, %Self.as_type.loc5_14.2 [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.1.%T (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref: type = name_ref T, @I.%T.loc4_13.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %t: @F.1.%T (%T) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc5_25.1: @I.%I.assoc_type (%I.assoc_type.1e5078.1) = assoc_entity element0, %F.decl [symbolic = %assoc0.loc5_25.2 (constants.%assoc0.8f0422.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %assoc0.loc5_25.1
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     witness = (%F.decl)
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic impl @impl(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type.loc9_21.2: type = facet_type <@I, @I(%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @impl.%I.type.loc9_21.2 (%I.type.325e65.2) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:   %impl_witness: <witness> = impl_witness (%F.decl), @impl(%U) [symbolic = %impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F.2, @impl(%U) [symbolic = %F.type (constants.%F.type.e88)]
// CHECK:STDOUT:   %F: @impl.%F.type (%F.type.e88) = struct_value () [symbolic = %F (constants.%F.b02)]
// CHECK:STDOUT:
// CHECK:STDOUT:   impl: %Self.ref as %I.type.loc9_21.1 {
// CHECK:STDOUT:     %F.decl: @impl.%F.type (%F.type.e88) = fn_decl @F.2 [symbolic = @impl.%F (constants.%F.b02)] {
// CHECK:STDOUT:       %self.patt: @F.2.%X (%X) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.2.%X (%X) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %t.patt: @F.2.%U (%U) = binding_pattern t
// CHECK:STDOUT:       %t.param_patt: @F.2.%U (%U) = value_param_pattern %t.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param: @F.2.%X (%X) = value_param call_param0
// CHECK:STDOUT:       %.loc10_16.1: type = splice_block %Self.ref [symbolic = %X (constants.%X)] {
// CHECK:STDOUT:         %.loc10_16.2: type = specific_constant constants.%X, @X(constants.%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:         %Self.ref: type = name_ref Self, %.loc10_16.2 [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @F.2.%X (%X) = bind_name self, %self.param
// CHECK:STDOUT:       %t.param: @F.2.%U (%U) = value_param call_param1
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %t: @F.2.%U (%U) = bind_name t, %t.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .U = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     witness = @X.%impl_witness
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @X(%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U.loc8_9.2: type = bind_symbolic_name U, 0 [symbolic = %U.loc8_9.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc8_9.2: type = symbolic_binding_pattern U, 0 [symbolic = %U.patt.loc8_9.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U.loc8_9.2)> [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @X.%I.type (%I.type.325e65.2) [symbolic = %require_complete (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     impl_decl @impl [concrete] {} {
// CHECK:STDOUT:       %Self.ref: type = name_ref Self, constants.%X [symbolic = %X (constants.%X)]
// CHECK:STDOUT:       %I.ref: %I.type.dac = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %U.ref: type = name_ref U, @X.%U.loc8_9.1 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:       %I.type.loc9_21.1: type = facet_type <@I, @I(constants.%U)> [symbolic = %I.type.loc9_21.2 (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %impl_witness: <witness> = impl_witness (@impl.%F.decl), @impl(constants.%U) [symbolic = @impl.%impl_witness (constants.%impl_witness)]
// CHECK:STDOUT:     %.loc9: type = specific_constant @impl.%I.type.loc9_21.1, @impl(constants.%U) [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
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
// CHECK:STDOUT: generic fn @F.1(@I.%T.loc4_13.1: type, @I.%Self.1: @I.%I.type (%I.type.325e65.1)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%T)> [symbolic = %I.type (constants.%I.type.325e65.1)]
// CHECK:STDOUT:   %Self: @F.1.%I.type (%I.type.325e65.1) = bind_symbolic_name Self, 1 [symbolic = %Self (constants.%Self.209206.1)]
// CHECK:STDOUT:   %Self.as_type.loc5_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc5_14.1 (constants.%Self.as_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.1.%Self.as_type.loc5_14.1 (%Self.as_type)](%t.param_patt: @F.1.%T (%T));
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(@X.%U.loc8_9.1: type) {
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 0 [symbolic = %U (constants.%U)]
// CHECK:STDOUT:   %X: type = class_type @X, @X(%U) [symbolic = %X (constants.%X)]
// CHECK:STDOUT:   %I.type: type = facet_type <@I, @I(%U)> [symbolic = %I.type (constants.%I.type.325e65.2)]
// CHECK:STDOUT:   %require_complete.loc10_25: <witness> = require_complete_type @F.2.%I.type (%I.type.325e65.2) [symbolic = %require_complete.loc10_25 (constants.%require_complete.cfe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_14: <witness> = require_complete_type @F.2.%X (%X) [symbolic = %require_complete.loc10_14 (constants.%require_complete.441)]
// CHECK:STDOUT:   %require_complete.loc10_23: <witness> = require_complete_type @F.2.%U (%U) [symbolic = %require_complete.loc10_23 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.2.%X (%X)](%t.param_patt: @F.2.%U (%U)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%T, constants.%Self.209206.1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.1
// CHECK:STDOUT:   %Self => constants.%Self.209206.1
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%Self.as_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.1.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(constants.%U) {
// CHECK:STDOUT:   %U.loc8_9.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc8_9.2 => constants.%U.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%U) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%U
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T.patt
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self.2 => constants.%Self.209206.2
// CHECK:STDOUT:   %F.type => constants.%F.type.2aef59.2
// CHECK:STDOUT:   %F => constants.%F.bb2dd4.2
// CHECK:STDOUT:   %I.assoc_type => constants.%I.assoc_type.1e5078.2
// CHECK:STDOUT:   %assoc0.loc5_25.2 => constants.%assoc0.8f0422.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type.loc9_21.2 => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete => constants.%require_complete.cfe
// CHECK:STDOUT:   %impl_witness => constants.%impl_witness
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type.e88
// CHECK:STDOUT:   %F => constants.%F.b02
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@impl.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@impl.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @impl(%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%U) {
// CHECK:STDOUT:   %U => constants.%U
// CHECK:STDOUT:   %X => constants.%X
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %require_complete.loc10_25 => constants.%require_complete.cfe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @X(@F.2.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.2.%U) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.1(constants.%U, constants.%I.facet) {
// CHECK:STDOUT:   %T => constants.%U
// CHECK:STDOUT:   %I.type => constants.%I.type.325e65.2
// CHECK:STDOUT:   %Self => constants.%I.facet
// CHECK:STDOUT:   %Self.as_type.loc5_14.1 => constants.%X
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@X.%U.loc8_9.2) {}
// CHECK:STDOUT:
