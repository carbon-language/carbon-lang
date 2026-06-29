// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/builtin/call_from_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/builtin/call_from_operator.carbon

// --- core.carbon

package Core;

fn MakeIntLiteral() -> type = "int_literal.make_type";
alias IntLiteral = MakeIntLiteral();
fn Int(N: IntLiteral) -> type = "int.make_type_signed";

interface AddWith(T:! type) {
  fn Op(self, other: Self) -> Self;
}

interface As(T:! type) {
  fn Convert(self) -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert(self) -> T;
}

impl i32 as AddWith(i32) {
  fn Op(self, other: Self) -> Self = "int.sadd";
}

impl IntLiteral as As(i32) {
  fn Convert(self) -> i32 = "int.convert_checked";
}

impl IntLiteral as ImplicitAs(i32) {
  fn Convert(self) -> i32 = "int.convert_checked";
}

impl i32 as ImplicitAs(IntLiteral) {
  fn Convert(self) -> IntLiteral = "int.convert_checked";
}

// --- user.carbon

import Core;

var arr: array(i32, (1 as i32) + (2 as i32)) = (3, 4, (3 as i32) + (4 as i32));


// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %return.param_patt.70e: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.3ae: %pattern_type.98f = return_slot_pattern %return.param_patt.70e, type [concrete]
// CHECK:STDOUT:   %MakeIntLiteral.type: type = fn_type @MakeIntLiteral [concrete]
// CHECK:STDOUT:   %MakeIntLiteral: %MakeIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = at_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.2ef: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.2ef = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.7e9: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.8d5: %AddWith.type.7e9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.48c: type = facet_access_type %Self.8d5 [symbolic]
// CHECK:STDOUT:   %pattern_type.131: type = pattern_type %Self.as_type.48c [symbolic]
// CHECK:STDOUT:   %self.param_patt.72e: %pattern_type.131 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.ac5: %pattern_type.131 = at_binding_pattern self, %self.param_patt.72e [symbolic]
// CHECK:STDOUT:   %other.param_patt.89e: %pattern_type.131 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %other.patt.f3d: %pattern_type.131 = at_binding_pattern other, %other.param_patt.89e [symbolic]
// CHECK:STDOUT:   %.e80: Core.Form = init_form %Self.as_type.48c [symbolic]
// CHECK:STDOUT:   %return.param_patt.8c6: %pattern_type.131 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.169: %pattern_type.131 = return_slot_pattern %return.param_patt.8c6, %Self.as_type.48c [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.e49: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.8d5) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.adc: %AddWith.WithSelf.Op.type.e49 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.588: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.aed: %AddWith.assoc_type.588 = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.c5f: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f80: %As.type.c5f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.1ce: type = facet_access_type %Self.f80 [symbolic]
// CHECK:STDOUT:   %pattern_type.a01: type = pattern_type %Self.as_type.1ce [symbolic]
// CHECK:STDOUT:   %self.param_patt.29a: %pattern_type.a01 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.a95: %pattern_type.a01 = at_binding_pattern self, %self.param_patt.29a [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.640: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.f80) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.896: %As.WithSelf.Convert.type.640 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.b40: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.a44: %As.assoc_type.b40 = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.bb1: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.5b8: type = facet_access_type %Self.bb1 [symbolic]
// CHECK:STDOUT:   %pattern_type.701: type = pattern_type %Self.as_type.5b8 [symbolic]
// CHECK:STDOUT:   %self.param_patt.c3f: %pattern_type.701 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.f5a: %pattern_type.701 = at_binding_pattern self, %self.param_patt.c3f [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.920: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.bb1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.293: %ImplicitAs.WithSelf.Convert.type.920 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.313: %ImplicitAs.assoc_type.fcc = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.7c6: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @i32.builtin.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.9ab: %AddWith.type.7c6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.aad: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.8d5) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.e15: %AddWith.WithSelf.Op.type.aad = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.439: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.7f8: %AddWith.assoc_type.439 = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %self.param_patt.936: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.703: %pattern_type.956 = at_binding_pattern self, %self.param_patt.936 [concrete]
// CHECK:STDOUT:   %other.param_patt.36d: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.760: %pattern_type.956 = at_binding_pattern other, %other.param_patt.36d [concrete]
// CHECK:STDOUT:   %.9cb: Core.Form = init_form %i32.builtin [concrete]
// CHECK:STDOUT:   %return.param_patt.903: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.930: %pattern_type.956 = return_slot_pattern %return.param_patt.903, %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.7c6 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.2dc: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.921: %AddWith.WithSelf.Op.type.2dc = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.dfb: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness @Core.IntLiteral.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.2c0: %As.type.dfb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.d9c: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.f80) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.140: %As.WithSelf.Convert.type.d9c = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.7f2: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.402: %As.assoc_type.7f2 = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %self.param_patt.1ca: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.cea: %pattern_type.dc0 = at_binding_pattern self, %self.param_patt.1ca [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dfb = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.e08: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.5a5: %As.WithSelf.Convert.type.e08 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.9df: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.300: <witness> = impl_witness @Core.IntLiteral.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.19a: %ImplicitAs.type.9df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.247: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.bb1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.e9a: %ImplicitAs.WithSelf.Convert.type.247 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.081: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.0d1: %ImplicitAs.assoc_type.081 = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.18f: %ImplicitAs.type.9df = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.300) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.20b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.18f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.b9f: %ImplicitAs.WithSelf.Convert.type.20b = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.9d7: <witness> = impl_witness @i32.builtin.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.c40: %ImplicitAs.type.7cb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.46f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.bb1) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.a78: %ImplicitAs.WithSelf.Convert.type.46f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e73: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.a37: %ImplicitAs.assoc_type.e73 = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %return.param_patt.eab: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.cd4: %pattern_type.dc0 = return_slot_pattern %return.param_patt.eab, Core.IntLiteral [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.776: %ImplicitAs.type.7cb = facet_value %i32.builtin, (%ImplicitAs.impl_witness.9d7) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.dc2: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.776) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.c73: %ImplicitAs.WithSelf.Convert.type.dc2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .MakeIntLiteral = %MakeIntLiteral.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeIntLiteral.decl: %MakeIntLiteral.type = fn_decl @MakeIntLiteral [concrete = constants.%MakeIntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete = constants.%return.param_patt.70e]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc4_24.1 [concrete = constants.%return.patt.3ae]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc4_24.2: Core.Form = init_form %.loc4_24.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeIntLiteral.ref: %MakeIntLiteral.type = name_ref MakeIntLiteral, %MakeIntLiteral.decl [concrete = constants.%MakeIntLiteral]
// CHECK:STDOUT:   %MakeIntLiteral.call: init type = call %MakeIntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc5_35.1: type = value_of_initializer %MakeIntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %.loc5_35.2: type = converted %MakeIntLiteral.call, %.loc5_35.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %IntLiteral: type = alias_binding IntLiteral, %.loc5_35.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%N.param_patt]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = at_binding_pattern N, %N.param_patt [concrete = constants.%N.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete = constants.%return.param_patt.70e]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc6_26.1 [concrete = constants.%return.patt.3ae]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_26.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc6_26.2: Core.Form = init_form %.loc6_26.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %N: Core.IntLiteral = wrapper_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.2ef = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt.loc8_20.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_23.1: type = splice_block %.loc8_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc8_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc8_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.116 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt.loc12_15.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc12_18.1: type = splice_block %.loc12_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc12_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc12_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.0ff = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc16_23.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc16_26.1: type = splice_block %.loc16_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc16_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc16_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.AddWith.impl [concrete] {} {
// CHECK:STDOUT:     %.loc20_6: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.2ef = name_ref AddWith, file.%AddWith.decl [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %.loc20_21: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.7c6]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.As.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.116 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %.loc24: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.dfb]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %.loc28: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.9df]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %.loc32: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.7cb]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc8_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc8_20.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc8_20.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc8_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc8_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc8_20.1)> [symbolic = %AddWith.type (constants.%AddWith.type.7e9)]
// CHECK:STDOUT:   %Self.loc8_29.2: @AddWith.%AddWith.type (%AddWith.type.7e9) = symbolic_binding Self, 1 [symbolic = %Self.loc8_29.2 (constants.%Self.8d5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc8_29.1: @AddWith.%AddWith.type (%AddWith.type.7e9) = symbolic_binding Self, 1 [symbolic = %Self.loc8_29.2 (constants.%Self.8d5)]
// CHECK:STDOUT:     %AddWith.WithSelf.decl = interface_with_self_decl @AddWith [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %AddWith.WithSelf.Op.decl: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.e49) = fn_decl @AddWith.WithSelf.Op [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.adc)] {
// CHECK:STDOUT:       %self.param_patt.loc9_9.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = value_param_pattern [symbolic = %self.param_patt.loc9_9.2 (constants.%self.param_patt.72e)]
// CHECK:STDOUT:       %self.patt.loc9_9.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = at_binding_pattern self, %self.param_patt.loc9_9.1 [symbolic = %self.patt.loc9_9.2 (constants.%self.patt.ac5)]
// CHECK:STDOUT:       %other.param_patt.loc9_20.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = value_param_pattern [symbolic = %other.param_patt.loc9_20.2 (constants.%other.param_patt.89e)]
// CHECK:STDOUT:       %other.patt.loc9_20.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = at_binding_pattern other, %other.param_patt.loc9_20.1 [symbolic = %other.patt.loc9_20.2 (constants.%other.patt.f3d)]
// CHECK:STDOUT:       %return.param_patt.loc9_31.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = out_param_pattern [symbolic = %return.param_patt.loc9_31.2 (constants.%return.param_patt.8c6)]
// CHECK:STDOUT:       %return.patt.loc9_28.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = return_slot_pattern %return.param_patt.loc9_31.1, %.loc9_31.3 [symbolic = %return.patt.loc9_28.2 (constants.%return.patt.169)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc9_31.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = specific_constant @AddWith.%Self.loc8_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:       %Self.ref.loc9_31: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = name_ref Self, %.loc9_31.2 [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:       %Self.as_type.loc9_31: type = facet_access_type %Self.ref.loc9_31 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:       %.loc9_31.3: type = converted %Self.ref.loc9_31, %Self.as_type.loc9_31 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:       %.loc9_31.4: Core.Form = init_form %.loc9_31.3 [symbolic = %.loc9_31.1 (constants.%.e80)]
// CHECK:STDOUT:       %self.param: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = value_param call_param0
// CHECK:STDOUT:       %.loc9_9.1: type = splice_block %.loc9_9.3 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)] {
// CHECK:STDOUT:         %.loc9_9.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = specific_constant @AddWith.%Self.loc8_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:         %Self.ref.loc9_9: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = name_ref Self, %.loc9_9.2 [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:         %Self.as_type.loc9_9.2: type = facet_access_type %Self.ref.loc9_9 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:         %.loc9_9.3: type = converted %Self.ref.loc9_9, %Self.as_type.loc9_9.2 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = value_param call_param1
// CHECK:STDOUT:       %.loc9_22.1: type = splice_block %.loc9_22.3 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)] {
// CHECK:STDOUT:         %.loc9_22.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = specific_constant @AddWith.%Self.loc8_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:         %Self.ref.loc9_22: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = name_ref Self, %.loc9_22.2 [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:         %Self.as_type.loc9_22: type = facet_access_type %Self.ref.loc9_22 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:         %.loc9_22.3: type = converted %Self.ref.loc9_22, %Self.as_type.loc9_22 [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = wrapper_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc9_35.1: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.588) = assoc_entity element0, %AddWith.WithSelf.Op.decl [symbolic = %assoc0.loc9_35.2 (constants.%assoc0.aed)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc8_29.1
// CHECK:STDOUT:     .Op = @AddWith.WithSelf.%assoc0.loc9_35.1
// CHECK:STDOUT:     witness = (@AddWith.WithSelf.%AddWith.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc12_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc12_15.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc12_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc12_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc12_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc12_15.1)> [symbolic = %As.type (constants.%As.type.c5f)]
// CHECK:STDOUT:   %Self.loc12_24.2: @As.%As.type (%As.type.c5f) = symbolic_binding Self, 1 [symbolic = %Self.loc12_24.2 (constants.%Self.f80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc12_24.1: @As.%As.type (%As.type.c5f) = symbolic_binding Self, 1 [symbolic = %Self.loc12_24.2 (constants.%Self.f80)]
// CHECK:STDOUT:     %As.WithSelf.decl = interface_with_self_decl @As [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %As.WithSelf.Convert.decl: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.640) = fn_decl @As.WithSelf.Convert [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.896)] {
// CHECK:STDOUT:       %self.param_patt.loc13_14.1: @As.WithSelf.Convert.%pattern_type.loc13_14 (%pattern_type.a01) = value_param_pattern [symbolic = %self.param_patt.loc13_14.2 (constants.%self.param_patt.29a)]
// CHECK:STDOUT:       %self.patt.loc13_14.1: @As.WithSelf.Convert.%pattern_type.loc13_14 (%pattern_type.a01) = at_binding_pattern self, %self.param_patt.loc13_14.1 [symbolic = %self.patt.loc13_14.2 (constants.%self.patt.a95)]
// CHECK:STDOUT:       %return.param_patt.loc13_23.1: @As.WithSelf.Convert.%pattern_type.loc13_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc13_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc13_20.1: @As.WithSelf.Convert.%pattern_type.loc13_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc13_23.1, %T.ref [symbolic = %return.patt.loc13_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc12_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc13_23.2: Core.Form = init_form %T.ref [symbolic = %.loc13_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @As.WithSelf.Convert.%Self.as_type.loc13_14.1 (%Self.as_type.1ce) = value_param call_param0
// CHECK:STDOUT:       %.loc13_14.1: type = splice_block %.loc13_14.3 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.1ce)] {
// CHECK:STDOUT:         %.loc13_14.2: @As.WithSelf.Convert.%As.type (%As.type.c5f) = specific_constant @As.%Self.loc12_24.1, @As(constants.%T) [symbolic = %Self (constants.%Self.f80)]
// CHECK:STDOUT:         %Self.ref: @As.WithSelf.Convert.%As.type (%As.type.c5f) = name_ref Self, %.loc13_14.2 [symbolic = %Self (constants.%Self.f80)]
// CHECK:STDOUT:         %Self.as_type.loc13_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.1ce)]
// CHECK:STDOUT:         %.loc13_14.3: type = converted %Self.ref, %Self.as_type.loc13_14.2 [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.1ce)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.WithSelf.Convert.%Self.as_type.loc13_14.1 (%Self.as_type.1ce) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc13_24.1: @As.WithSelf.%As.assoc_type (%As.assoc_type.b40) = assoc_entity element0, %As.WithSelf.Convert.decl [symbolic = %assoc0.loc13_24.2 (constants.%assoc0.a44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc12_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @As.WithSelf.%assoc0.loc13_24.1
// CHECK:STDOUT:     witness = (@As.WithSelf.%As.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc16_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc16_23.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc16_23.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc16_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc16_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc16_23.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self.loc16_32.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc16_32.2 (constants.%Self.bb1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc16_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self.loc16_32.2 (constants.%Self.bb1)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.920) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.293)] {
// CHECK:STDOUT:       %self.param_patt.loc17_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_14 (%pattern_type.701) = value_param_pattern [symbolic = %self.param_patt.loc17_14.2 (constants.%self.param_patt.c3f)]
// CHECK:STDOUT:       %self.patt.loc17_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_14 (%pattern_type.701) = at_binding_pattern self, %self.param_patt.loc17_14.1 [symbolic = %self.patt.loc17_14.2 (constants.%self.patt.f5a)]
// CHECK:STDOUT:       %return.param_patt.loc17_23.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc17_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc17_20.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc17_23.1, %T.ref [symbolic = %return.patt.loc17_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc16_23.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc17_23.2: Core.Form = init_form %T.ref [symbolic = %.loc17_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc17_14.1 (%Self.as_type.5b8) = value_param call_param0
// CHECK:STDOUT:       %.loc17_14.1: type = splice_block %.loc17_14.3 [symbolic = %Self.as_type.loc17_14.1 (constants.%Self.as_type.5b8)] {
// CHECK:STDOUT:         %.loc17_14.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = specific_constant @ImplicitAs.%Self.loc16_32.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.bb1)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = name_ref Self, %.loc17_14.2 [symbolic = %Self (constants.%Self.bb1)]
// CHECK:STDOUT:         %Self.as_type.loc17_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc17_14.1 (constants.%Self.as_type.5b8)]
// CHECK:STDOUT:         %.loc17_14.3: type = converted %Self.ref, %Self.as_type.loc17_14.2 [symbolic = %Self.as_type.loc17_14.1 (constants.%Self.as_type.5b8)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc17_14.1 (%Self.as_type.5b8) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc17_24.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc17_24.2 (constants.%assoc0.313)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc16_32.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc17_24.1
// CHECK:STDOUT:     .IntLiteral = <poisoned>
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: %.loc20_6 as %AddWith.type {
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.decl: %i32.builtin.as.AddWith.impl.Op.type = fn_decl @i32.builtin.as.AddWith.impl.Op [concrete = constants.%i32.builtin.as.AddWith.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%self.param_patt.936]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.703]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%other.param_patt.36d]
// CHECK:STDOUT:     %other.patt: %pattern_type.956 = at_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.760]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %Self.ref.loc21_31 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc21_31: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc20_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc21: Core.Form = init_form %Self.ref.loc21_31 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc21_9: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc20_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc21_22: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc20_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = wrapper_binding other, %other.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param2
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%i32.builtin.as.AddWith.impl.Op.decl), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness %AddWith.impl_witness_table [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Op = %i32.builtin.as.AddWith.impl.Op.decl
// CHECK:STDOUT:   extend %AddWith.type
// CHECK:STDOUT:   witness = %AddWith.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: %IntLiteral.ref as %As.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.decl: %Core.IntLiteral.as.As.impl.Convert.type = fn_decl @Core.IntLiteral.as.As.impl.Convert [concrete = constants.%Core.IntLiteral.as.As.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%self.param_patt.1ca]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cea]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc25_23.1 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc25_23.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc25_23.2: Core.Form = init_form %.loc25_23.1 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.As.impl.%IntLiteral.ref [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.As.impl.Convert.decl), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness %As.impl_witness_table [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.As.impl.Convert.decl
// CHECK:STDOUT:   extend %As.type
// CHECK:STDOUT:   witness = %As.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: %IntLiteral.ref as %ImplicitAs.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = fn_decl @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%self.param_patt.1ca]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.cea]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc29_23.1 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc29_23.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc29_23.2: Core.Form = init_form %.loc29_23.1 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.ImplicitAs.impl.%IntLiteral.ref [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.300]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: %.loc32 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.decl: %i32.builtin.as.ImplicitAs.impl.Convert.type = fn_decl @i32.builtin.as.ImplicitAs.impl.Convert [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%self.param_patt.936]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = at_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.703]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete = constants.%return.param_patt.eab]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %IntLiteral.ref [concrete = constants.%return.patt.cd4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc33: Core.Form = init_form %IntLiteral.ref [concrete = constants.%.f7d]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @i32.builtin.as.ImplicitAs.impl.%.loc32 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.9d7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.builtin.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @MakeIntLiteral() -> out %return.param: type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(@AddWith.%T.loc8_20.2: type, @AddWith.%Self.loc8_29.1: @AddWith.%AddWith.type (%AddWith.type.7e9)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.7e9)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.8d5)]
// CHECK:STDOUT:   %Self.as_type.loc9_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc9_9.1 (constants.%Self.as_type.48c)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc9_9.1 [symbolic = %pattern_type (constants.%pattern_type.131)]
// CHECK:STDOUT:   %self.param_patt.loc9_9.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = value_param_pattern [symbolic = %self.param_patt.loc9_9.2 (constants.%self.param_patt.72e)]
// CHECK:STDOUT:   %self.patt.loc9_9.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = at_binding_pattern self, %self.param_patt.loc9_9.2 [symbolic = %self.patt.loc9_9.2 (constants.%self.patt.ac5)]
// CHECK:STDOUT:   %other.param_patt.loc9_20.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = value_param_pattern [symbolic = %other.param_patt.loc9_20.2 (constants.%other.param_patt.89e)]
// CHECK:STDOUT:   %other.patt.loc9_20.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = at_binding_pattern other, %other.param_patt.loc9_20.2 [symbolic = %other.patt.loc9_20.2 (constants.%other.patt.f3d)]
// CHECK:STDOUT:   %.loc9_31.1: Core.Form = init_form %Self.as_type.loc9_9.1 [symbolic = %.loc9_31.1 (constants.%.e80)]
// CHECK:STDOUT:   %return.param_patt.loc9_31.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = out_param_pattern [symbolic = %return.param_patt.loc9_31.2 (constants.%return.param_patt.8c6)]
// CHECK:STDOUT:   %return.patt.loc9_28.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.131) = return_slot_pattern %return.param_patt.loc9_31.2, %Self.as_type.loc9_9.1 [symbolic = %return.patt.loc9_28.2 (constants.%return.patt.169)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c), %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c)) -> out %return.param: @AddWith.WithSelf.Op.%Self.as_type.loc9_9.1 (%Self.as_type.48c);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(@As.%T.loc12_15.2: type, @As.%Self.loc12_24.1: @As.%As.type (%As.type.c5f)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.c5f)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.c5f) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f80)]
// CHECK:STDOUT:   %Self.as_type.loc13_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc13_14.1 (constants.%Self.as_type.1ce)]
// CHECK:STDOUT:   %pattern_type.loc13_14: type = pattern_type %Self.as_type.loc13_14.1 [symbolic = %pattern_type.loc13_14 (constants.%pattern_type.a01)]
// CHECK:STDOUT:   %self.param_patt.loc13_14.2: @As.WithSelf.Convert.%pattern_type.loc13_14 (%pattern_type.a01) = value_param_pattern [symbolic = %self.param_patt.loc13_14.2 (constants.%self.param_patt.29a)]
// CHECK:STDOUT:   %self.patt.loc13_14.2: @As.WithSelf.Convert.%pattern_type.loc13_14 (%pattern_type.a01) = at_binding_pattern self, %self.param_patt.loc13_14.2 [symbolic = %self.patt.loc13_14.2 (constants.%self.patt.a95)]
// CHECK:STDOUT:   %.loc13_23.1: Core.Form = init_form %T [symbolic = %.loc13_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc13_23: type = pattern_type %T [symbolic = %pattern_type.loc13_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc13_23.2: @As.WithSelf.Convert.%pattern_type.loc13_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc13_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc13_20.2: @As.WithSelf.Convert.%pattern_type.loc13_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc13_23.2, %T [symbolic = %return.patt.loc13_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.WithSelf.Convert.%Self.as_type.loc13_14.1 (%Self.as_type.1ce)) -> out %return.param: @As.WithSelf.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%T.loc16_23.2: type, @ImplicitAs.%Self.loc16_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.bb1)]
// CHECK:STDOUT:   %Self.as_type.loc17_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc17_14.1 (constants.%Self.as_type.5b8)]
// CHECK:STDOUT:   %pattern_type.loc17_14: type = pattern_type %Self.as_type.loc17_14.1 [symbolic = %pattern_type.loc17_14 (constants.%pattern_type.701)]
// CHECK:STDOUT:   %self.param_patt.loc17_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_14 (%pattern_type.701) = value_param_pattern [symbolic = %self.param_patt.loc17_14.2 (constants.%self.param_patt.c3f)]
// CHECK:STDOUT:   %self.patt.loc17_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_14 (%pattern_type.701) = at_binding_pattern self, %self.param_patt.loc17_14.2 [symbolic = %self.patt.loc17_14.2 (constants.%self.patt.f5a)]
// CHECK:STDOUT:   %.loc17_23.1: Core.Form = init_form %T [symbolic = %.loc17_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc17_23: type = pattern_type %T [symbolic = %pattern_type.loc17_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc17_23.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc17_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc17_20.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc17_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc17_23.2, %T [symbolic = %return.patt.loc17_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc17_14.1 (%Self.as_type.5b8)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op(%self.param: %i32.builtin, %other.param: %i32.builtin) -> out %return.param: %i32.builtin = "int.sadd";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert(%self.param: Core.IntLiteral) -> out %return.param: %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.ImplicitAs.impl.Convert(%self.param: Core.IntLiteral) -> out %return.param: %i32.builtin = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.ImplicitAs.impl.Convert(%self.param: %i32.builtin) -> out %return.param: Core.IntLiteral = "int.convert_checked";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc8_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.8d5) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.8d5) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7e9
// CHECK:STDOUT:   %Self => constants.%Self.8d5
// CHECK:STDOUT:   %Self.as_type.loc9_9.1 => constants.%Self.as_type.48c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.131
// CHECK:STDOUT:   %self.param_patt.loc9_9.2 => constants.%self.param_patt.72e
// CHECK:STDOUT:   %self.patt.loc9_9.2 => constants.%self.patt.ac5
// CHECK:STDOUT:   %other.param_patt.loc9_20.2 => constants.%other.param_patt.89e
// CHECK:STDOUT:   %other.patt.loc9_20.2 => constants.%other.patt.f3d
// CHECK:STDOUT:   %.loc9_31.1 => constants.%.e80
// CHECK:STDOUT:   %return.param_patt.loc9_31.2 => constants.%return.param_patt.8c6
// CHECK:STDOUT:   %return.patt.loc9_28.2 => constants.%return.patt.169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.f80) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.f80) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.c5f
// CHECK:STDOUT:   %Self => constants.%Self.f80
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => constants.%Self.as_type.1ce
// CHECK:STDOUT:   %pattern_type.loc13_14 => constants.%pattern_type.a01
// CHECK:STDOUT:   %self.param_patt.loc13_14.2 => constants.%self.param_patt.29a
// CHECK:STDOUT:   %self.patt.loc13_14.2 => constants.%self.patt.a95
// CHECK:STDOUT:   %.loc13_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc13_23 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc13_23.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc13_20.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc16_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc16_23.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.bb1) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.bb1) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.bb1
// CHECK:STDOUT:   %Self.as_type.loc17_14.1 => constants.%Self.as_type.5b8
// CHECK:STDOUT:   %pattern_type.loc17_14 => constants.%pattern_type.701
// CHECK:STDOUT:   %self.param_patt.loc17_14.2 => constants.%self.param_patt.c3f
// CHECK:STDOUT:   %self.patt.loc17_14.2 => constants.%self.patt.f5a
// CHECK:STDOUT:   %.loc17_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc17_23 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc17_23.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc17_20.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc8_20.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc8_20.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self.loc8_29.2 => constants.%Self.9ab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.8d5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%Self.8d5
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.aad
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.e15
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.439
// CHECK:STDOUT:   %assoc0.loc9_35.2 => constants.%assoc0.7f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.2dc
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.921
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.439
// CHECK:STDOUT:   %assoc0.loc9_35.2 => constants.%assoc0.7f8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.as_type.loc9_9.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT:   %self.param_patt.loc9_9.2 => constants.%self.param_patt.936
// CHECK:STDOUT:   %self.patt.loc9_9.2 => constants.%self.patt.703
// CHECK:STDOUT:   %other.param_patt.loc9_20.2 => constants.%other.param_patt.36d
// CHECK:STDOUT:   %other.patt.loc9_20.2 => constants.%other.patt.760
// CHECK:STDOUT:   %.loc9_31.1 => constants.%.9cb
// CHECK:STDOUT:   %return.param_patt.loc9_31.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc9_28.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc12_15.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc12_15.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self.loc12_24.2 => constants.%Self.2c0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.f80) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%Self.f80
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.d9c
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.140
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.7f2
// CHECK:STDOUT:   %assoc0.loc13_24.2 => constants.%assoc0.402
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.e08
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.5a5
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.7f2
// CHECK:STDOUT:   %assoc0.loc13_24.2 => constants.%assoc0.402
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc13_14.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc13_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %self.param_patt.loc13_14.2 => constants.%self.param_patt.1ca
// CHECK:STDOUT:   %self.patt.loc13_14.2 => constants.%self.patt.cea
// CHECK:STDOUT:   %.loc13_23.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc13_23 => constants.%pattern_type.956
// CHECK:STDOUT:   %return.param_patt.loc13_23.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc13_20.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc16_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc16_23.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self.loc16_32.2 => constants.%Self.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.bb1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%Self.bb1
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.247
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.e9a
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.081
// CHECK:STDOUT:   %assoc0.loc17_24.2 => constants.%assoc0.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.18f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.18f
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.20b
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.b9f
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.081
// CHECK:STDOUT:   %assoc0.loc17_24.2 => constants.%assoc0.0d1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.18f) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.18f
// CHECK:STDOUT:   %Self.as_type.loc17_14.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc17_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %self.param_patt.loc17_14.2 => constants.%self.param_patt.1ca
// CHECK:STDOUT:   %self.patt.loc17_14.2 => constants.%self.patt.cea
// CHECK:STDOUT:   %.loc17_23.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc17_23 => constants.%pattern_type.956
// CHECK:STDOUT:   %return.param_patt.loc17_23.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc17_20.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.patt.loc16_23.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc16_23.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self.loc16_32.2 => constants.%Self.c40
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.bb1) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%Self.bb1
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.46f
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.a78
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.e73
// CHECK:STDOUT:   %assoc0.loc17_24.2 => constants.%assoc0.a37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.776) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.776
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.dc2
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.c73
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.e73
// CHECK:STDOUT:   %assoc0.loc17_24.2 => constants.%assoc0.a37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.776) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.776
// CHECK:STDOUT:   %Self.as_type.loc17_14.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type.loc17_14 => constants.%pattern_type.956
// CHECK:STDOUT:   %self.param_patt.loc17_14.2 => constants.%self.param_patt.936
// CHECK:STDOUT:   %self.patt.loc17_14.2 => constants.%self.patt.703
// CHECK:STDOUT:   %.loc17_23.1 => constants.%.f7d
// CHECK:STDOUT:   %pattern_type.loc17_23 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %return.param_patt.loc17_23.2 => constants.%return.param_patt.eab
// CHECK:STDOUT:   %return.patt.loc17_20.2 => constants.%return.patt.cd4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.c5f: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.37a: %As.type.c5f = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.b40: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ff8: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.37a) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.87d: %As.WithSelf.Convert.type.ff8 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type.d4a: type = facet_access_type %Self.37a [symbolic]
// CHECK:STDOUT:   %pattern_type.e33: type = pattern_type %Self.as_type.d4a [symbolic]
// CHECK:STDOUT:   %self.param_patt.9e9: %pattern_type.e33 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.51c: %pattern_type.e33 = at_binding_pattern self, %self.param_patt.9e9 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %As.type.dfb: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.256: %As.type.dfb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.675: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.37a) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.dd6: %As.WithSelf.Convert.type.675 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.7f2: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.656: %As.assoc_type.7f2 = assoc_entity element0, imports.%Core.import_ref.967 [concrete]
// CHECK:STDOUT:   %assoc0.4ec: %As.assoc_type.b40 = assoc_entity element0, imports.%Core.import_ref.47a [symbolic]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.dfb = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dbd: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.82a: %As.WithSelf.Convert.type.dbd = struct_value () [concrete]
// CHECK:STDOUT:   %.161: type = fn_type_with_self_type %As.WithSelf.Convert.type.dbd, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.959: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.2ef: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.2ef = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.7e9: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.96a: %AddWith.type.7e9 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.588: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.28e: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.96a) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.eab: %AddWith.WithSelf.Op.type.28e = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.331: type = facet_access_type %Self.96a [symbolic]
// CHECK:STDOUT:   %pattern_type.aed: type = pattern_type %Self.as_type.331 [symbolic]
// CHECK:STDOUT:   %return.param_patt.7c8: %pattern_type.aed = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.1c8: %pattern_type.aed = return_slot_pattern %return.param_patt.7c8, invalid [symbolic]
// CHECK:STDOUT:   %self.param_patt.aae: %pattern_type.aed = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.c4d: %pattern_type.aed = at_binding_pattern self, %self.param_patt.aae [symbolic]
// CHECK:STDOUT:   %.4cc: Core.Form = init_form %Self.as_type.331 [symbolic]
// CHECK:STDOUT:   %other.param_patt.076: %pattern_type.aed = value_param_pattern [symbolic]
// CHECK:STDOUT:   %other.patt.667: %pattern_type.aed = at_binding_pattern other, %other.param_patt.076 [symbolic]
// CHECK:STDOUT:   %AddWith.type.7c6: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.bd2: %AddWith.type.7c6 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.0ff: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.96a) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.19f: %AddWith.WithSelf.Op.type.0ff = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.439: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.e3c: %AddWith.assoc_type.439 = assoc_entity element0, imports.%Core.import_ref.3cf [concrete]
// CHECK:STDOUT:   %assoc0.912: %AddWith.assoc_type.588 = assoc_entity element0, imports.%Core.import_ref.976 [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.7c6 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.43a: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.128: %AddWith.WithSelf.Op.type.43a = struct_value () [concrete]
// CHECK:STDOUT:   %.e4a: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.43a, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.351: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.3aa: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.294: %ImplicitAs.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.fcc: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.97e: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.945: %ImplicitAs.WithSelf.Convert.type.97e = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.ef0: type = facet_access_type %Self.294 [symbolic]
// CHECK:STDOUT:   %pattern_type.20e: type = pattern_type %Self.as_type.ef0 [symbolic]
// CHECK:STDOUT:   %self.param_patt.360: %pattern_type.20e = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.25f: %pattern_type.20e = at_binding_pattern self, %self.param_patt.360 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.7cb: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.f63: %ImplicitAs.type.7cb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.e70: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.792: %ImplicitAs.WithSelf.Convert.type.e70 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.e73: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.625: %ImplicitAs.assoc_type.e73 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %assoc0.0ac9: %ImplicitAs.assoc_type.fcc = assoc_entity element0, imports.%Core.import_ref.f9d [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.9df: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.a70: %ImplicitAs.type.9df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.750: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.5cf [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a82: %ImplicitAs.type.7cb = facet_value %i32.builtin, (%ImplicitAs.impl_witness.750) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7af: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.a82) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.a32: %ImplicitAs.WithSelf.Convert.type.7af = struct_value () [concrete]
// CHECK:STDOUT:   %.5b0: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.7af, %ImplicitAs.facet.a82 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9e2: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %arr.patt: %pattern_type.9e2 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:   %arr.var_patt: %pattern_type.9e2 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.94b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.0f9: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.c75: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_3.1ba, %int_4.0c1, %int_7) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.51c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.294) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.78c: %ImplicitAs.WithSelf.Convert.type.51c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.081: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.da6: %ImplicitAs.assoc_type.081 = assoc_entity element0, imports.%Core.import_ref.cb2 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.da4: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.dfd [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.9b9: %ImplicitAs.type.9df = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.da4) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.4fb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.9b9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.838: %ImplicitAs.WithSelf.Convert.type.4fb = struct_value () [concrete]
// CHECK:STDOUT:   %.b73: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.4fb, %ImplicitAs.facet.9b9 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.94b: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0f9: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %array: %array_type = tuple_value (%int_3.a0f, %int_4.4f1, %int_7) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .As = %Core.As
// CHECK:STDOUT:     .AddWith = %Core.AddWith
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//default, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//default, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.e36: @As.WithSelf.%As.assoc_type (%As.assoc_type.b40) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%assoc0 (constants.%assoc0.4ec)]
// CHECK:STDOUT:   %Core.Convert.b64 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.967: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.ff8) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.87d)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.d2cec9.2: @As.%As.type (%As.type.c5f) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%Self (constants.%Self.37a)]
// CHECK:STDOUT:   %Core.import_ref.3f4 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.47a = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.ed7: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.2dc10a.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.908: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.type.dfb]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.2ef = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.562: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.588) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc0 (constants.%assoc0.912)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.3cf: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.28e) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.eab)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.5: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.4c6974.2: @AddWith.%AddWith.type (%AddWith.type.7e9) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%Self (constants.%Self.96a)]
// CHECK:STDOUT:   %Core.import_ref.763 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.976 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.37d: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.4ee: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.type.7c6]
// CHECK:STDOUT:   %Core.import_ref.9c3: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.484: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.fcc) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.0ac9)]
// CHECK:STDOUT:   %Core.Convert.368 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.cb2: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.97e) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.945)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.8: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.f19eaa.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.294)]
// CHECK:STDOUT:   %Core.import_ref.8c8 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.9: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.f9d = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.eab: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.da4]
// CHECK:STDOUT:   %Core.import_ref.2dc10a.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.7e0: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.9df]
// CHECK:STDOUT:   %Core.import_ref.f49: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.750]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.017: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.7cb]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.5cf = impl_witness_table (%Core.import_ref.06c923.2), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.dfd = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.9e2 = ref_binding_pattern arr [concrete = constants.%arr.patt]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.9e2 = var_pattern %arr.patt [concrete = constants.%arr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_44: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc4_16: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc4_27: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.161 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.5d7]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc4_40: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.161 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.959]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.e4a = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.351]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.5b0 = impl_witness_access constants.%ImplicitAs.impl_witness.750, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %i32.builtin.as.AddWith.impl.Op.call, %impl.elem0.loc4_32.2 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %i32.builtin.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %i32.builtin.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_32.4, %.loc4_16 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = wrapper_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.c5f)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.c5f) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.37a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.3f4
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.e36
// CHECK:STDOUT:     witness = (imports.%Core.Convert.b64)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.b3bc94.6: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.7e9)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.7e9) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.96a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.763
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.562
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.9: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.294)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.8c8
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.484
// CHECK:STDOUT:     witness = (imports.%Core.Convert.368)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.2dc10a.1 as imports.%Core.import_ref.908 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.ed7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.63cbb1.2 as imports.%Core.import_ref.4ee [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.37d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.2dc10a.2 as imports.%Core.import_ref.7e0 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.eab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.63cbb1.3 as imports.%Core.import_ref.017 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.f49
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.d2cec9.2: @As.%As.type (%As.type.c5f)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.c5f)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.c5f) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.37a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.d4a)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.e33)]
// CHECK:STDOUT:   %self.param_patt: @As.WithSelf.Convert.%pattern_type.1 (%pattern_type.e33) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.9e9)]
// CHECK:STDOUT:   %self.patt: @As.WithSelf.Convert.%pattern_type.1 (%pattern_type.e33) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.51c)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @As.WithSelf.Convert.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @As.WithSelf.Convert.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(imports.%Core.import_ref.b3bc94.5: type, imports.%Core.import_ref.4c6974.2: @AddWith.%AddWith.type (%AddWith.type.7e9)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.7e9)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.7e9) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.96a)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.331)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.aed)]
// CHECK:STDOUT:   %self.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.aae)]
// CHECK:STDOUT:   %self.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.c4d)]
// CHECK:STDOUT:   %other.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = value_param_pattern [symbolic = %other.param_patt (constants.%other.param_patt.076)]
// CHECK:STDOUT:   %other.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = at_binding_pattern other, %other.param_patt [symbolic = %other.patt (constants.%other.patt.667)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Self.as_type [symbolic = %.1 (constants.%.4cc)]
// CHECK:STDOUT:   %return.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.7c8)]
// CHECK:STDOUT:   %return.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.aed) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.1c8)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.8: type, imports.%Core.import_ref.f19eaa.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.3aa)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.3aa)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.3aa) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.294)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.ef0)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.20e)]
// CHECK:STDOUT:   %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.1 (%pattern_type.20e) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.360)]
// CHECK:STDOUT:   %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.1 (%pattern_type.20e) = at_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.25f)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.2 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.2 (%pattern_type.51d) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.c39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.ImplicitAs.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.ImplicitAs.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3.loc4_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc4_52: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_3.loc4_56: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc4_61: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.161 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.94b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc4_74: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.161 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.0f9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.e4a = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.c75]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.b73 = impl_witness_access constants.%ImplicitAs.impl_witness.da4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.94b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin to %.loc4_78.3 = in_place_init %.loc4_78.2 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.b73 = impl_witness_access constants.%ImplicitAs.impl_witness.da4, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.2: <bound method> = bound_method %int_4.loc4_52, %impl.elem0.loc4_78.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.0f9]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.2: init %i32.builtin = call %bound_method.loc4_78.2(%int_4.loc4_52) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_78.5: init %i32.builtin = converted %int_4.loc4_52, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.2 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:   %.loc4_78.6: ref %i32.builtin = array_index file.%arr.var, %int_1
// CHECK:STDOUT:   %.loc4_78.7: init %i32.builtin to %.loc4_78.6 = in_place_init %.loc4_78.5 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:   %.loc4_78.8: ref %i32.builtin = array_index file.%arr.var, %int_2
// CHECK:STDOUT:   %.loc4_78.9: init %i32.builtin to %.loc4_78.8 = in_place_init %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.10: init %array_type to file.%arr.var = array_init (%.loc4_78.4, %.loc4_78.7, %.loc4_78.9) [concrete = constants.%array]
// CHECK:STDOUT:   %.loc4_1: init %array_type = converted %.loc4_78.1, %.loc4_78.10 [concrete = constants.%array]
// CHECK:STDOUT:   assign file.%arr.var, %.loc4_1
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.37a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.37a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.c5f
// CHECK:STDOUT:   %Self => constants.%Self.37a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.d4a
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.e33
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.9e9
// CHECK:STDOUT:   %self.patt => constants.%self.patt.51c
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%Self.256
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.37a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%Self.37a
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.675
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.dd6
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.7f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.dfb
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.dbd
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.82a
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.7f2
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.656
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.96a) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.96a) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7e9
// CHECK:STDOUT:   %Self => constants.%Self.96a
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.331
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.aed
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.aae
// CHECK:STDOUT:   %self.patt => constants.%self.patt.c4d
// CHECK:STDOUT:   %other.param_patt => constants.%other.param_patt.076
// CHECK:STDOUT:   %other.patt => constants.%other.patt.667
// CHECK:STDOUT:   %.1 => constants.%.4cc
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.7c8
// CHECK:STDOUT:   %return.patt => constants.%return.patt.1c8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%Self.bd2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.96a) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%Self.96a
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.0ff
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.19f
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.439
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.7c6
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.43a
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.128
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.439
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.e3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.294) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.294) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.ef0
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.20e
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.360
// CHECK:STDOUT:   %self.patt => constants.%self.patt.25f
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%Self.f63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.294) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.e70
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.792
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.e73
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.625
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%Self.a70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.a82) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.7cb
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.a82
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.7af
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.a32
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.e73
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.625
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.294) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%Self.294
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.51c
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.78c
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.081
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.da6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.9b9) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.9df
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.9b9
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.4fb
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.838
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.081
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.da6
// CHECK:STDOUT: }
// CHECK:STDOUT:
