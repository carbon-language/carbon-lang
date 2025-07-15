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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %pattern_type.7dcd0a.1: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Cfn.type: type = fn_type @Cfn [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Cfn: %Cfn.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.type: type = fn_type @H [concrete]
// CHECK:STDOUT:   %H: %H.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %H.specific_fn.1ed: <specific function> = specific_function %H, @H(%T) [symbolic]
// CHECK:STDOUT:   %Cfn.specific_fn.53f: <specific function> = specific_function %Cfn, @Cfn(%T) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.153: type = fn_type @Op.1, @Destroy.impl(%C) [concrete]
// CHECK:STDOUT:   %Op.5d7: %Op.type.153 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %Op.specific_fn.6eb: <specific function> = specific_function %Op.5d7, @Op.1(%C) [concrete]
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.2, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%To) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.1, @ImplicitAs.impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.1(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.specific_fn.501: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%i32) [concrete]
// CHECK:STDOUT:   %Op.type.a17: type = fn_type @Op.1, @Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Op.e6a: %Op.type.a17 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.235: type = ptr_type %i32 [concrete]
// CHECK:STDOUT:   %Op.specific_fn.014: <specific function> = specific_function %Op.e6a, @Op.1(%i32) [concrete]
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
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @ImplicitAs.impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/parts/int, loc16_39, loaded [symbolic = @ImplicitAs.impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @ImplicitAs.impl.4f9 [concrete]
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
// CHECK:STDOUT:     %T.loc20_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc20_6.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc20_6.2 [symbolic = %T.loc20_6.1 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc20_6.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @H.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc23_25: type = name_ref T, %T.loc23_6.2 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc23_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @H.%T.loc23_6.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc23_19: type = name_ref T, %T.loc23_6.2 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:     %x: @H.%T.loc23_6.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @H.%T.loc23_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%T.loc23_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @G.%pattern_type (%pattern_type.7dcd0a.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc29_25: type = name_ref T, %T.loc29_6.2 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:     %T.loc29_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @G.%T.loc29_6.1 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc29_19: type = name_ref T, %T.loc29_6.2 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:     %x: @G.%T.loc29_6.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc29_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc29_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
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
// CHECK:STDOUT:     %T.loc16_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_22.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Cfn.%T.loc16_22.2 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_22.1 [symbolic = %T.loc16_22.2 (constants.%T)]
// CHECK:STDOUT:     %x: @Cfn.%T.loc16_22.2 (%T) = bind_name x, %x.param
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
// CHECK:STDOUT: generic fn @Cfn(%T.loc16_22.1: type) {
// CHECK:STDOUT:   %T.loc16_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_22.2 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc16_22.2 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc16_22.2 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: %C, %x.param: @Cfn.%T.loc16_22.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc20_6.2: type) {
// CHECK:STDOUT:   %T.loc20_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc20_6.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc20_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc20_6.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc20_6.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc23_6.2: type) {
// CHECK:STDOUT:   %T.loc23_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc23_6.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc23_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc23_6.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %F.specific_fn.loc25_3.2: <specific function> = specific_function constants.%F, @F(%T.loc23_6.1) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @H.%T.loc23_6.1 (%T)) -> @H.%T.loc23_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc25: @H.%T.loc23_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc25_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc25_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc25_3.1(%x.ref.loc25)
// CHECK:STDOUT:     %x.ref.loc26: @H.%T.loc23_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref.loc26
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc29_6.2: type) {
// CHECK:STDOUT:   %T.loc29_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc29_6.1 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc29_6.1 [symbolic = %pattern_type (constants.%pattern_type.7dcd0a.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc29_6.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %H.specific_fn.loc30_3.2: <specific function> = specific_function constants.%H, @H(%T.loc29_6.1) [symbolic = %H.specific_fn.loc30_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:   %F.specific_fn.loc31_3.2: <specific function> = specific_function constants.%F, @F(%T.loc29_6.1) [symbolic = %F.specific_fn.loc31_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:   %Cfn.specific_fn.loc34_4.2: <specific function> = specific_function constants.%Cfn, @Cfn(%T.loc29_6.1) [symbolic = %Cfn.specific_fn.loc34_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @G.%T.loc29_6.1 (%T)) -> @G.%T.loc29_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:     %x.ref.loc30: @G.%T.loc29_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %H.specific_fn.loc30_3.1: <specific function> = specific_function %H.ref, @H(constants.%T) [symbolic = %H.specific_fn.loc30_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:     %H.call: init @G.%T.loc29_6.1 (%T) = call %H.specific_fn.loc30_3.1(%x.ref.loc30)
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc31: @G.%T.loc29_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc31_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc31_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc31_3.1(%x.ref.loc31)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %pattern_type.c48 = binding_pattern c [concrete]
// CHECK:STDOUT:       %c.var_patt: %pattern_type.c48 = var_pattern %c.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref %C = var %c.var_patt
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %Cfn.ref: %Cfn.type = name_ref Cfn, @C.%Cfn.decl [concrete = constants.%Cfn]
// CHECK:STDOUT:     %Cfn.bound: <bound method> = bound_method %c.ref, %Cfn.ref
// CHECK:STDOUT:     %x.ref.loc34: @G.%T.loc29_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Cfn.specific_fn.loc34_4.1: <specific function> = specific_function %Cfn.ref, @Cfn(constants.%T) [symbolic = %Cfn.specific_fn.loc34_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:     %bound_method.loc34: <bound method> = bound_method %c.ref, %Cfn.specific_fn.loc34_4.1
// CHECK:STDOUT:     %.loc34: %C = bind_value %c.ref
// CHECK:STDOUT:     %Cfn.call: init %empty_tuple.type = call %bound_method.loc34(%.loc34, %x.ref.loc34)
// CHECK:STDOUT:     %x.ref.loc35: @G.%T.loc29_6.1 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Op.bound: <bound method> = bound_method %c.var, constants.%Op.5d7
// CHECK:STDOUT:     %Op.specific_fn: <specific function> = specific_function constants.%Op.5d7, @Op.1(constants.%C) [concrete = constants.%Op.specific_fn.6eb]
// CHECK:STDOUT:     %bound_method.loc33: <bound method> = bound_method %c.var, %Op.specific_fn
// CHECK:STDOUT:     %addr: %ptr.019 = addr_of %c.var
// CHECK:STDOUT:     %no_op: init %empty_tuple.type = call %bound_method.loc33(%addr)
// CHECK:STDOUT:     return %x.ref.loc35
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
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc39_3.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.1(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc39_3.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc39_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc39_3: init %i32 = converted %int_0, %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc39_3
// CHECK:STDOUT:   %.loc39_10: type = splice_block %i32.loc39 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %pattern_type.7ce = binding_pattern m [concrete]
// CHECK:STDOUT:     %m.var_patt: %pattern_type.7ce = var_pattern %m.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i32 = var %m.var_patt
// CHECK:STDOUT:   %.loc40: type = splice_block %i32.loc40 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc40: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc40: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i32 = bind_name m, %m.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %n.ref.loc42: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn.501]
// CHECK:STDOUT:   %.loc42: %i32 = bind_value %n.ref.loc42
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc42)
// CHECK:STDOUT:   %m.ref: ref %i32 = name_ref m, %m
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %n.ref.loc43: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc43: %i32 = bind_value %n.ref.loc43
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn(%.loc43)
// CHECK:STDOUT:   assign %m.ref, %G.call
// CHECK:STDOUT:   %Op.bound.loc40: <bound method> = bound_method %m.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.1: <specific function> = specific_function constants.%Op.e6a, @Op.1(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc40: <bound method> = bound_method %m.var, %Op.specific_fn.1
// CHECK:STDOUT:   %addr.loc40: %ptr.235 = addr_of %m.var
// CHECK:STDOUT:   %no_op.loc40: init %empty_tuple.type = call %bound_method.loc40(%addr.loc40)
// CHECK:STDOUT:   %Op.bound.loc39: <bound method> = bound_method %n.var, constants.%Op.e6a
// CHECK:STDOUT:   %Op.specific_fn.2: <specific function> = specific_function constants.%Op.e6a, @Op.1(constants.%i32) [concrete = constants.%Op.specific_fn.014]
// CHECK:STDOUT:   %bound_method.loc39_3.3: <bound method> = bound_method %n.var, %Op.specific_fn.2
// CHECK:STDOUT:   %addr.loc39: %ptr.235 = addr_of %n.var
// CHECK:STDOUT:   %no_op.loc39: init %empty_tuple.type = call %bound_method.loc39_3.3(%addr.loc39)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%T) {
// CHECK:STDOUT:   %T.loc16_22.2 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc20_6.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T) {
// CHECK:STDOUT:   %T.loc23_6.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %F.specific_fn.loc25_3.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc29_6.1 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dcd0a.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.loc20_6.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %T.loc29_6.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %H.specific_fn.loc30_3.2 => constants.%H.specific_fn.aac
// CHECK:STDOUT:   %F.specific_fn.loc31_3.2 => constants.%F.specific_fn.501
// CHECK:STDOUT:   %Cfn.specific_fn.loc34_4.2 => constants.%Cfn.specific_fn.7b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32) {
// CHECK:STDOUT:   %T.loc23_6.1 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %F.specific_fn.loc25_3.2 => constants.%F.specific_fn.501
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%i32) {
// CHECK:STDOUT:   %T.loc16_22.2 => constants.%i32
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7ce
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
