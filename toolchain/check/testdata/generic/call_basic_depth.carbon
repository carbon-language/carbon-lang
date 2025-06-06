// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/generic/call_basic_depth.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/generic/call_basic_depth.carbon

class C {
  fn Cfn[self: Self, T:! type](x: T) {
  }
}

fn F[T:! type](x: T) {
}

fn H[T:! type](x: T) -> T {
  // TODO: Definition should be emitted for H for this to work in lowering.
  F(x);
  return x;
}

fn G[T:! type](x: T) -> T {
  H(x);
  F(x);

  var c: C;
  c.Cfn(x);
  return x;
}

fn M() {
  var n: i32 = 0;
  var m: i32;

  F(n);
  m = G(n);
}

// CHECK:STDOUT: --- call_basic_depth.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %Cfn.type: type = fn_type @Cfn [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cfn: %Cfn.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T.8b3) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.specific_fn.1ed: <specific function> = specific_function %H, @H(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.906: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.2, @impl.49c(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.494: %Destroy.type = facet_value %C, (%Destroy.impl_witness.906) [concrete]
// CHECK:STDOUT:   %.b80: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.494 [concrete]
// CHECK:STDOUT:   %Cfn.specific_fn.53f: <specific function> = specific_function %Cfn, @Cfn(%T.8b3) [symbolic]
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M.e98: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.64e: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.2, @impl.49c(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.facet.5f7: %Destroy.type = facet_value %i32, (%Destroy.impl_witness.64e) [concrete]
// CHECK:STDOUT:   %.ab7: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet.5f7 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.921: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet.921 [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.specific_fn.501: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%i32) [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.2(%i32) [concrete]
// CHECK:STDOUT:   %H.specific_fn.aac: <specific function> = specific_function %H, @H(%i32) [concrete]
// CHECK:STDOUT:   %Cfn.specific_fn.7b2: <specific function> = specific_function %Cfn, @Cfn(%i32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/destroy, loc15_23, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x.param: @F.%T.loc19_6.2 (%T.8b3) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x: @F.%T.loc19_6.2 (%T.8b3) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc22_25: type = name_ref T, %T.loc22_6.1 [symbolic = %T.loc22_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %T.loc22_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x.param: @H.%T.loc22_6.2 (%T.8b3) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc22_19: type = name_ref T, %T.loc22_6.1 [symbolic = %T.loc22_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x: @H.%T.loc22_6.2 (%T.8b3) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @H.%T.loc22_6.2 (%T.8b3) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%T.loc22_6.2 (%T.8b3) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc28_25: type = name_ref T, %T.loc28_6.1 [symbolic = %T.loc28_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %T.loc28_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc28_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x.param: @G.%T.loc28_6.2 (%T.8b3) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc28_19: type = name_ref T, %T.loc28_6.1 [symbolic = %T.loc28_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:     %x: @G.%T.loc28_6.2 (%T.8b3) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc28_6.2 (%T.8b3) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc28_6.2 (%T.8b3) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M.e98] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Cfn.decl: %Cfn.type = fn_decl @Cfn [concrete = constants.%Cfn] {
// CHECK:STDOUT:     %self.patt: %pattern_type.c48 = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: %pattern_type.c48 = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Cfn.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Cfn.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %T.loc15_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %x.param: @Cfn.%T.loc15_22.1 (%T.8b3) = value_param call_param1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc15_22.2 [symbolic = %T.loc15_22.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %x: @Cfn.%T.loc15_22.1 (%T.8b3) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Cfn = %Cfn.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Cfn(%T.loc15_22.2: type) {
// CHECK:STDOUT:   %T.loc15_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_22.1 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc15_22.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc15_22.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %C, %x.param: @Cfn.%T.loc15_22.1 (%T.8b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc19_6.2 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc19_6.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc19_6.2 (%T.8b3)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc22_6.1: type) {
// CHECK:STDOUT:   %T.loc22_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc22_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc22_6.2 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc22_6.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %F.specific_fn.loc24_3.2: <specific function> = specific_function constants.%F, @F(%T.loc22_6.2) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @H.%T.loc22_6.2 (%T.8b3)) -> @H.%T.loc22_6.2 (%T.8b3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc24: @H.%T.loc22_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc24_3.1: <specific function> = specific_function %F.ref, @F(constants.%T.8b3) [symbolic = %F.specific_fn.loc24_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc24_3.1(%x.ref.loc24)
// CHECK:STDOUT:     %x.ref.loc25: @H.%T.loc22_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref.loc25
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc28_6.1: type) {
// CHECK:STDOUT:   %T.loc28_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc28_6.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc28_6.2 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc28_6.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %H.specific_fn.loc29_3.2: <specific function> = specific_function constants.%H, @H(%T.loc28_6.2) [symbolic = %H.specific_fn.loc29_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:   %F.specific_fn.loc30_3.2: <specific function> = specific_function constants.%F, @F(%T.loc28_6.2) [symbolic = %F.specific_fn.loc30_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:   %Cfn.specific_fn.loc33_4.2: <specific function> = specific_function constants.%Cfn, @Cfn(%T.loc28_6.2) [symbolic = %Cfn.specific_fn.loc33_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%T.loc28_6.2 (%T.8b3)) -> @G.%T.loc28_6.2 (%T.8b3) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:     %x.ref.loc29: @G.%T.loc28_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     %H.specific_fn.loc29_3.1: <specific function> = specific_function %H.ref, @H(constants.%T.8b3) [symbolic = %H.specific_fn.loc29_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:     %H.call: init @G.%T.loc28_6.2 (%T.8b3) = call %H.specific_fn.loc29_3.1(%x.ref.loc29)
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc30: @G.%T.loc28_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc30_3.1: <specific function> = specific_function %F.ref, @F(constants.%T.8b3) [symbolic = %F.specific_fn.loc30_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc30_3.1(%x.ref.loc30)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:       %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:     %impl.elem0: %.b80 = impl_witness_access constants.%Destroy.impl_witness.906, element0 [concrete = constants.%Op.5d7]
// CHECK:STDOUT:     %bound_method.loc32: <bound method> = bound_method %c.var, %impl.elem0
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %Cfn.ref: %Cfn.type = name_ref Cfn, @C.%Cfn.decl [concrete = constants.%Cfn]
// CHECK:STDOUT:     %Cfn.bound: <bound method> = bound_method %c.ref, %Cfn.ref
// CHECK:STDOUT:     %x.ref.loc33: @G.%T.loc28_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     %Cfn.specific_fn.loc33_4.1: <specific function> = specific_function %Cfn.ref, @Cfn(constants.%T.8b3) [symbolic = %Cfn.specific_fn.loc33_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:     %bound_method.loc33: <bound method> = bound_method %c.ref, %Cfn.specific_fn.loc33_4.1
// CHECK:STDOUT:     %.loc33: %C = bind_value %c.ref
// CHECK:STDOUT:     %Cfn.call: init %empty_tuple.type = call %bound_method.loc33(%.loc33, %x.ref.loc33)
// CHECK:STDOUT:     %x.ref.loc34: @G.%T.loc28_6.2 (%T.8b3) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref.loc34
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %pattern_type.7ce = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.var_patt: %pattern_type.7ce = var_pattern %n.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var %n.var_patt
// CHECK:STDOUT:   %impl.elem0.loc38_3.1: %.ab7 = impl_witness_access constants.%Destroy.impl_witness.64e, element0 [concrete = constants.%Op.e6a]
// CHECK:STDOUT:   %bound_method.loc38_3.1: <bound method> = bound_method %n.var, %impl.elem0.loc38_3.1
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0.loc38_3.2: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc38_3.2: <bound method> = bound_method %int_0, %impl.elem0.loc38_3.2 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn.loc38_3.1: <specific function> = specific_function %impl.elem0.loc38_3.2, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc38_3.3: <bound method> = bound_method %int_0, %specific_fn.loc38_3.1 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc38_3.3(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc38_3.1: init %i32 = converted %int_0, %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc38_3.1
// CHECK:STDOUT:   %.loc38_10: type = splice_block %i32.loc38 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = binding_pattern m [concrete]
// CHECK:STDOUT:     %m.var_patt: %pattern_type.7ce = var_pattern %m.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i32 = var %m.var_patt
// CHECK:STDOUT:   %impl.elem0.loc39: %.ab7 = impl_witness_access constants.%Destroy.impl_witness.64e, element0 [concrete = constants.%Op.e6a]
// CHECK:STDOUT:   %bound_method.loc39: <bound method> = bound_method %m.var, %impl.elem0.loc39
// CHECK:STDOUT:   %.loc39_10: type = splice_block %i32.loc39 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i32 = bind_name m, %m.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %n.ref.loc41: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn.501]
// CHECK:STDOUT:   %.loc41: %i32 = bind_value %n.ref.loc41
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc41)
// CHECK:STDOUT:   %m.ref: ref %i32 = name_ref m, %m
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %n.ref.loc42: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc42: %i32 = bind_value %n.ref.loc42
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn(%.loc42)
// CHECK:STDOUT:   assign %m.ref, %G.call
// CHECK:STDOUT:   %specific_fn.loc39: <specific function> = specific_function %impl.elem0.loc39, @Op.2(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.1: <bound method> = bound_method %m.var, %specific_fn.loc39
// CHECK:STDOUT:   %.loc39_3: %i32 = bind_value %m.var
// CHECK:STDOUT:   %no_op.1: init %empty_tuple.type = call %bound_method.1(%.loc39_3)
// CHECK:STDOUT:   %specific_fn.loc38_3.2: <specific function> = specific_function %impl.elem0.loc38_3.1, @Op.2(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.2: <bound method> = bound_method %n.var, %specific_fn.loc38_3.2
// CHECK:STDOUT:   %.loc38_3.2: %i32 = bind_value %n.var
// CHECK:STDOUT:   %no_op.2: init %empty_tuple.type = call %bound_method.2(%.loc38_3.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc22_6.2 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %F.specific_fn.loc24_3.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc28_6.2 => constants.%T.8b3
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %T.loc28_6.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %H.specific_fn.loc29_3.2 => constants.%H.specific_fn.aac
// CHECK:STDOUT:   %F.specific_fn.loc30_3.2 => constants.%F.specific_fn.501
// CHECK:STDOUT:   %Cfn.specific_fn.loc33_4.2 => constants.%Cfn.specific_fn.7b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32) {
// CHECK:STDOUT:   %T.loc22_6.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn.loc24_3.2 => constants.%F.specific_fn.501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%i32) {
// CHECK:STDOUT:   %T.loc15_22.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
