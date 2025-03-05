// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %M.type: type = fn_type @M [concrete]
// CHECK:STDOUT:   %M: %M.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %complete_type.f8a: <witness> = complete_type_witness %i32.builtin [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %impl_witness.d39: <witness> = impl_witness (imports.%Core.import_ref.a5b), @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, %impl_witness.d39 [concrete]
// CHECK:STDOUT:   %.a0b: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %F.specific_fn.501: <specific function> = specific_function %F, @F(%i32) [concrete]
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G, @G(%i32) [concrete]
// CHECK:STDOUT:   %H.specific_fn.aac: <specific function> = specific_function %H, @H(%i32) [concrete]
// CHECK:STDOUT:   %Cfn.specific_fn.7b2: <specific function> = specific_function %Cfn, @Cfn(%i32) [concrete]
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
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .H = %H.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .M = %M.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc16_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @F.%T.loc16_6.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%T.loc16_6.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc16_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @F.%T.loc16_6.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc16_6.1 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:     %x: @F.%T.loc16_6.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl: %H.type = fn_decl @H [concrete = constants.%H] {
// CHECK:STDOUT:     %T.patt.loc19_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @H.%T.loc19_6.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @H.%T.loc19_6.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @H.%T.loc19_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @H.%T.loc19_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19_25: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc19_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @H.%T.loc19_6.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc19_19: type = name_ref T, %T.loc19_6.1 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:     %x: @H.%T.loc19_6.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @H.%T.loc19_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @H.%T.loc19_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %T.patt.loc25_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @G.%T.loc25_6.2 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @G.%T.loc25_6.2 (%T) = value_param_pattern %x.patt, call_param0
// CHECK:STDOUT:     %return.patt: @G.%T.loc25_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%T.loc25_6.2 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc25_25: type = name_ref T, %T.loc25_6.1 [symbolic = %T.loc25_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc25_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc25_6.2 (constants.%T)]
// CHECK:STDOUT:     %x.param: @G.%T.loc25_6.2 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc25_19: type = name_ref T, %T.loc25_6.1 [symbolic = %T.loc25_6.2 (constants.%T)]
// CHECK:STDOUT:     %x: @G.%T.loc25_6.2 (%T) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @G.%T.loc25_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @G.%T.loc25_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %M.decl: %M.type = fn_decl @M [concrete = constants.%M] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %Cfn.decl: %Cfn.type = fn_decl @Cfn [concrete = constants.%Cfn] {
// CHECK:STDOUT:     %self.patt: %C = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: %C = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %T.patt.loc12_22.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_22.2 (constants.%T.patt)]
// CHECK:STDOUT:     %x.patt: @Cfn.%T.loc12_22.1 (%T) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @Cfn.%T.loc12_22.1 (%T) = value_param_pattern %x.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: %C = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, constants.%C [concrete = constants.%C]
// CHECK:STDOUT:     %self: %C = bind_name self, %self.param
// CHECK:STDOUT:     %T.loc12_22.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_22.1 (constants.%T)]
// CHECK:STDOUT:     %x.param: @Cfn.%T.loc12_22.1 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc12_22.2 [symbolic = %T.loc12_22.1 (constants.%T)]
// CHECK:STDOUT:     %x: @Cfn.%T.loc12_22.1 (%T) = bind_name x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT:   .Cfn = %Cfn.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Cfn(%T.loc12_22.2: type) {
// CHECK:STDOUT:   %T.loc12_22.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc12_22.1 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc12_22.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_22.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Cfn.%T.loc12_22.1 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: %C, %T.patt.loc12_22.1: type](%x.param_patt: @Cfn.%T.loc12_22.1 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc16_6.1: type) {
// CHECK:STDOUT:   %T.loc16_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc16_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc16_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc16_6.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc16_6.1: type](%x.param_patt: @F.%T.loc16_6.2 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @H(%T.loc19_6.1: type) {
// CHECK:STDOUT:   %T.loc19_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc19_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc19_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @H.%T.loc19_6.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %F.specific_fn.loc21_3.2: <specific function> = specific_function constants.%F, @F(%T.loc19_6.2) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc19_6.1: type](%x.param_patt: @H.%T.loc19_6.2 (%T)) -> @H.%T.loc19_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc21: @H.%T.loc19_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc21_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc21_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc21_3.1(%x.ref.loc21)
// CHECK:STDOUT:     %x.ref.loc22: @H.%T.loc19_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref.loc22
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(%T.loc25_6.1: type) {
// CHECK:STDOUT:   %T.loc25_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc25_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc25_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc25_6.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @G.%T.loc25_6.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %H.specific_fn.loc26_3.2: <specific function> = specific_function constants.%H, @H(%T.loc25_6.2) [symbolic = %H.specific_fn.loc26_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:   %F.specific_fn.loc27_3.2: <specific function> = specific_function constants.%F, @F(%T.loc25_6.2) [symbolic = %F.specific_fn.loc27_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:   %Cfn.specific_fn.loc30_4.2: <specific function> = specific_function constants.%Cfn, @Cfn(%T.loc25_6.2) [symbolic = %Cfn.specific_fn.loc30_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.patt.loc25_6.1: type](%x.param_patt: @G.%T.loc25_6.2 (%T)) -> @G.%T.loc25_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %H.ref: %H.type = name_ref H, file.%H.decl [concrete = constants.%H]
// CHECK:STDOUT:     %x.ref.loc26: @G.%T.loc25_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %H.specific_fn.loc26_3.1: <specific function> = specific_function %H.ref, @H(constants.%T) [symbolic = %H.specific_fn.loc26_3.2 (constants.%H.specific_fn.1ed)]
// CHECK:STDOUT:     %H.call: init @G.%T.loc25_6.2 (%T) = call %H.specific_fn.loc26_3.1(%x.ref.loc26)
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %x.ref.loc27: @G.%T.loc25_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %F.specific_fn.loc27_3.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc27_3.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init %empty_tuple.type = call %F.specific_fn.loc27_3.1(%x.ref.loc27)
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %c.patt: %C = binding_pattern c
// CHECK:STDOUT:       %.loc29: %C = var_pattern %c.patt
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %c.var: ref %C = var c
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %c: ref %C = bind_name c, %c.var
// CHECK:STDOUT:     %c.ref: ref %C = name_ref c, %c
// CHECK:STDOUT:     %Cfn.ref: %Cfn.type = name_ref Cfn, @C.%Cfn.decl [concrete = constants.%Cfn]
// CHECK:STDOUT:     %Cfn.bound: <bound method> = bound_method %c.ref, %Cfn.ref
// CHECK:STDOUT:     %x.ref.loc30: @G.%T.loc25_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     %Cfn.specific_fn.loc30_4.1: <specific function> = specific_function %Cfn.ref, @Cfn(constants.%T) [symbolic = %Cfn.specific_fn.loc30_4.2 (constants.%Cfn.specific_fn.53f)]
// CHECK:STDOUT:     %bound_method: <bound method> = bound_method %c.ref, %Cfn.specific_fn.loc30_4.1
// CHECK:STDOUT:     %.loc30: %C = bind_value %c.ref
// CHECK:STDOUT:     %Cfn.call: init %empty_tuple.type = call %bound_method(%.loc30, %x.ref.loc30)
// CHECK:STDOUT:     %x.ref.loc31: @G.%T.loc25_6.2 (%T) = name_ref x, %x
// CHECK:STDOUT:     return %x.ref.loc31
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @M() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %n.patt: %i32 = binding_pattern n
// CHECK:STDOUT:     %.loc35_3.1: %i32 = var_pattern %n.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n.var: ref %i32 = var n
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.a0b = impl_witness_access constants.%impl_witness.d39, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc35_3.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc35_3.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc35_3.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc35_3.2: init %i32 = converted %int_0, %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   assign %n.var, %.loc35_3.2
// CHECK:STDOUT:   %.loc35_10: type = splice_block %i32.loc35 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc35: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc35: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %n: ref %i32 = bind_name n, %n.var
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %m.patt: %i32 = binding_pattern m
// CHECK:STDOUT:     %.loc36_3: %i32 = var_pattern %m.patt
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m.var: ref %i32 = var m
// CHECK:STDOUT:   %.loc36_10: type = splice_block %i32.loc36 [concrete = constants.%i32] {
// CHECK:STDOUT:     %int_32.loc36: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc36: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %m: ref %i32 = bind_name m, %m.var
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %n.ref.loc38: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%i32) [concrete = constants.%F.specific_fn.501]
// CHECK:STDOUT:   %.loc38: %i32 = bind_value %n.ref.loc38
// CHECK:STDOUT:   %F.call: init %empty_tuple.type = call %F.specific_fn(%.loc38)
// CHECK:STDOUT:   %m.ref: ref %i32 = name_ref m, %m
// CHECK:STDOUT:   %G.ref: %G.type = name_ref G, file.%G.decl [concrete = constants.%G]
// CHECK:STDOUT:   %n.ref.loc39: ref %i32 = name_ref n, %n
// CHECK:STDOUT:   %G.specific_fn: <specific function> = specific_function %G.ref, @G(constants.%i32) [concrete = constants.%G.specific_fn]
// CHECK:STDOUT:   %.loc39: %i32 = bind_value %n.ref.loc39
// CHECK:STDOUT:   %G.call: init %i32 = call %G.specific_fn(%.loc39)
// CHECK:STDOUT:   assign %m.ref, %G.call
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%T) {
// CHECK:STDOUT:   %T.loc12_22.1 => constants.%T
// CHECK:STDOUT:   %T.patt.loc12_22.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc16_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc16_6.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%T) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc19_6.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %F.specific_fn.loc21_3.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@H.%T.loc19_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc25_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc25_6.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(@G.%T.loc25_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(@G.%T.loc25_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(@G.%T.loc25_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%i32) {
// CHECK:STDOUT:   %T.loc16_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc16_6.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%i32) {
// CHECK:STDOUT:   %T.loc25_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc25_6.2 => constants.%i32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.f8a
// CHECK:STDOUT:   %H.specific_fn.loc26_3.2 => constants.%H.specific_fn.aac
// CHECK:STDOUT:   %F.specific_fn.loc27_3.2 => constants.%F.specific_fn.501
// CHECK:STDOUT:   %Cfn.specific_fn.loc30_4.2 => constants.%Cfn.specific_fn.7b2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @H(constants.%i32) {
// CHECK:STDOUT:   %T.loc19_6.2 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc19_6.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Cfn(constants.%i32) {
// CHECK:STDOUT:   %T.loc12_22.1 => constants.%i32
// CHECK:STDOUT:   %T.patt.loc12_22.2 => constants.%i32
// CHECK:STDOUT: }
// CHECK:STDOUT:
