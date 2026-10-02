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
fn Array(T: type, N: IntLiteral) -> type = "array.make_type";

interface AddWith(T: type) {
  fn Op(self, other: Self) -> Self;
}

interface As(T: type) {
  fn Convert(self) -> T;
}

interface ImplicitAs(T: type) {
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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.576: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %return.param_patt.90b: %pattern_type.9a5 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.718: %pattern_type.9a5 = return_slot_pattern %return.param_patt.90b, type [concrete]
// CHECK:STDOUT:   %MakeIntLiteral.type: type = fn_type @MakeIntLiteral [concrete]
// CHECK:STDOUT:   %MakeIntLiteral: %MakeIntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %N.patt: %pattern_type.dc0 = wrapper_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.param_patt: %pattern_type.9a5 = value_param_pattern [concrete]
// CHECK:STDOUT:   %T.patt.a5b: %pattern_type.9a5 = wrapper_binding_pattern T, %T.param_patt [concrete]
// CHECK:STDOUT:   %Array.type: type = fn_type @Array [concrete]
// CHECK:STDOUT:   %Array: %Array.type = struct_value () [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.2ef: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.2ef = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.fe0: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.544: type = facet_access_type %Self.fe0 [symbolic]
// CHECK:STDOUT:   %pattern_type.458: type = pattern_type %Self.as_type.544 [symbolic]
// CHECK:STDOUT:   %self.param_patt.445: %pattern_type.458 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.740: %pattern_type.458 = wrapper_binding_pattern self, %self.param_patt.445 [symbolic]
// CHECK:STDOUT:   %other.param_patt.1dd: %pattern_type.458 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %other.patt.c41: %pattern_type.458 = wrapper_binding_pattern other, %other.param_patt.1dd [symbolic]
// CHECK:STDOUT:   %.581: Core.Form = init_form %Self.as_type.544 [symbolic]
// CHECK:STDOUT:   %return.param_patt.8f2: %pattern_type.458 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.46c: %pattern_type.458 = return_slot_pattern %return.param_patt.8f2, %Self.as_type.544 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.287: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.fe0) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.5ab: %AddWith.WithSelf.Op.type.287 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.728: %AddWith.assoc_type.1d5 = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.12c: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.b18: %As.type.12c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.5c1: type = facet_access_type %Self.b18 [symbolic]
// CHECK:STDOUT:   %pattern_type.f4a: type = pattern_type %Self.as_type.5c1 [symbolic]
// CHECK:STDOUT:   %self.param_patt.915: %pattern_type.f4a = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.3aa: %pattern_type.f4a = wrapper_binding_pattern self, %self.param_patt.915 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.40e: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.b18) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.d7e: %As.WithSelf.Convert.type.40e = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.e47: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.722: %As.assoc_type.e47 = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b02: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.757: %ImplicitAs.type.b02 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.0dd: type = facet_access_type %Self.757 [symbolic]
// CHECK:STDOUT:   %pattern_type.97b: type = pattern_type %Self.as_type.0dd [symbolic]
// CHECK:STDOUT:   %self.param_patt.163: %pattern_type.97b = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.da1: %pattern_type.97b = wrapper_binding_pattern self, %self.param_patt.163 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.1b3: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.757) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.6e2: %ImplicitAs.WithSelf.Convert.type.1b3 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.07d: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.0ee: %ImplicitAs.assoc_type.07d = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.8a5: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %.723: <witness> = impl_self_witness %i32.builtin, @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @i32.builtin.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.69c: %AddWith.type.8a5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.139: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.fe0) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.9b0: %AddWith.WithSelf.Op.type.139 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.5ec: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.33e: %AddWith.assoc_type.5ec = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %self.param_patt.936: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.16e: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt.936 [concrete]
// CHECK:STDOUT:   %other.param_patt.36d: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:   %other.patt.8f0: %pattern_type.956 = wrapper_binding_pattern other, %other.param_patt.36d [concrete]
// CHECK:STDOUT:   %.9cb: Core.Form = init_form %i32.builtin [concrete]
// CHECK:STDOUT:   %return.param_patt.903: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.930: %pattern_type.956 = return_slot_pattern %return.param_patt.903, %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.8a5 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.872: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.6e9: %AddWith.WithSelf.Op.type.872 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.294: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %.e40: <witness> = impl_self_witness Core.IntLiteral, @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness @Core.IntLiteral.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.1d5: %As.type.294 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.046: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.b18) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.e01: %As.WithSelf.Convert.type.046 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.12b: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.092: %As.assoc_type.12b = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %self.param_patt.1ca: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.d07: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt.1ca [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.294 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.c8a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.baa: %As.WithSelf.Convert.type.c8a = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb4: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %.985: <witness> = impl_self_witness Core.IntLiteral, @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c40: <witness> = impl_witness @Core.IntLiteral.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.0be: %ImplicitAs.type.cb4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.6f6: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.757) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.4a7: %ImplicitAs.WithSelf.Convert.type.6f6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.d73: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.a2d: %ImplicitAs.assoc_type.d73 = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.1e5: %ImplicitAs.type.cb4 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c40) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c1c: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.1e5) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.a8a: %ImplicitAs.WithSelf.Convert.type.c1c = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %.f5a: <witness> = impl_self_witness %i32.builtin, @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.780: <witness> = impl_witness @i32.builtin.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.ee1: %ImplicitAs.type.5be = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.0bc: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.757) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.67f: %ImplicitAs.WithSelf.Convert.type.0bc = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.32b: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.879: %ImplicitAs.assoc_type.32b = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %return.param_patt.eab: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.cd4: %pattern_type.dc0 = return_slot_pattern %return.param_patt.eab, Core.IntLiteral [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.69f: %ImplicitAs.type.5be = facet_value %i32.builtin, (%ImplicitAs.impl_witness.780) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.c18: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.69f) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.d73: %ImplicitAs.WithSelf.Convert.type.c18 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .MakeIntLiteral = %MakeIntLiteral.decl
// CHECK:STDOUT:     .IntLiteral = %IntLiteral
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .Array = %Array.decl
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %MakeIntLiteral.decl: %MakeIntLiteral.type = fn_decl @MakeIntLiteral [concrete = constants.%MakeIntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9a5 = out_param_pattern [concrete = constants.%return.param_patt.90b]
// CHECK:STDOUT:     %return.patt: %pattern_type.9a5 = return_slot_pattern %return.param_patt, %.loc4_24.1 [concrete = constants.%return.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_24.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc4_24.2: Core.Form = init_form %.loc4_24.1 [concrete = constants.%.576]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = wrapper_binding_pattern N, %N.param_patt [concrete = constants.%N.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9a5 = out_param_pattern [concrete = constants.%return.param_patt.90b]
// CHECK:STDOUT:     %return.patt: %pattern_type.9a5 = return_slot_pattern %return.param_patt, %.loc6_26.1 [concrete = constants.%return.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_26.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc6_26.2: Core.Form = init_form %.loc6_26.1 [concrete = constants.%.576]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %N: Core.IntLiteral = wrapper_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Array.decl: %Array.type = fn_decl @Array [concrete = constants.%Array] {
// CHECK:STDOUT:     %T.param_patt: %pattern_type.9a5 = value_param_pattern [concrete = constants.%T.param_patt]
// CHECK:STDOUT:     %T.patt: %pattern_type.9a5 = wrapper_binding_pattern T, %T.param_patt [concrete = constants.%T.patt.a5b]
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete = constants.%N.param_patt]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = wrapper_binding_pattern N, %N.param_patt [concrete = constants.%N.patt]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9a5 = out_param_pattern [concrete = constants.%return.param_patt.90b]
// CHECK:STDOUT:     %return.patt: %pattern_type.9a5 = return_slot_pattern %return.param_patt, %.loc7_37.1 [concrete = constants.%return.patt.718]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_37.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc7_37.2: Core.Form = init_form %.loc7_37.1 [concrete = constants.%.576]
// CHECK:STDOUT:     %T.param: type = value_param call_param0
// CHECK:STDOUT:     %.loc7_13: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %T: type = wrapper_binding T, %T.param
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param1
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %N: Core.IntLiteral = wrapper_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param2
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.2ef = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt.loc9_20.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_20.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc9_22.1: type = splice_block %.loc9_22.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc9_22.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc9_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.116 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt.loc13_15.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc13_17.1: type = splice_block %.loc13_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc13_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.0ff = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt.loc17_23.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_23.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_25.1: type = splice_block %.loc17_25.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc17_25.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc17_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.AddWith.impl [concrete] {} {
// CHECK:STDOUT:     %.loc21_6: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.2ef = name_ref AddWith, file.%AddWith.decl [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %.loc21_21: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.8a5]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc21: <witness> = impl_self_witness @i32.builtin.as.AddWith.impl.%.loc21_6, @AddWith, @AddWith(constants.%i32.builtin) [concrete = constants.%.723]
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.As.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.116 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %.loc25: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.294]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc25: <witness> = impl_self_witness @Core.IntLiteral.as.As.impl.%IntLiteral.ref, @As, @As(constants.%i32.builtin) [concrete = constants.%.e40]
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %.loc29: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.cb4]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc29: <witness> = impl_self_witness @Core.IntLiteral.as.ImplicitAs.impl.%IntLiteral.ref, @ImplicitAs, @ImplicitAs(constants.%i32.builtin) [concrete = constants.%.985]
// CHECK:STDOUT:   impl_decl @i32.builtin.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %.loc33: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.0ff = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.5be]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc33: <witness> = impl_self_witness @i32.builtin.as.ImplicitAs.impl.%.loc33, @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete = constants.%.f5a]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc9_20.2: type) {
// CHECK:STDOUT:   %T.patt.loc9_20.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc9_20.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc9_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc9_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc9_20.1)> [symbolic = %AddWith.type (constants.%AddWith.type.872)]
// CHECK:STDOUT:   %Self.loc9_28.2: @AddWith.%AddWith.type (%AddWith.type.872) = symbolic_binding Self, 1 [symbolic = %Self.loc9_28.2 (constants.%Self.fe0)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc9_28.1: @AddWith.%AddWith.type (%AddWith.type.872) = symbolic_binding Self, 1 [symbolic = %Self.loc9_28.2 (constants.%Self.fe0)]
// CHECK:STDOUT:     %AddWith.WithSelf.decl = interface_with_self_decl @AddWith [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %AddWith.WithSelf.Op.decl: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.287) = fn_decl @AddWith.WithSelf.Op [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.5ab)] {
// CHECK:STDOUT:       %self.param_patt.loc10_9.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = value_param_pattern [symbolic = %self.param_patt.loc10_9.2 (constants.%self.param_patt.445)]
// CHECK:STDOUT:       %self.patt.loc10_9.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = wrapper_binding_pattern self, %self.param_patt.loc10_9.1 [symbolic = %self.patt.loc10_9.2 (constants.%self.patt.740)]
// CHECK:STDOUT:       %other.param_patt.loc10_20.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = value_param_pattern [symbolic = %other.param_patt.loc10_20.2 (constants.%other.param_patt.1dd)]
// CHECK:STDOUT:       %other.patt.loc10_20.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = wrapper_binding_pattern other, %other.param_patt.loc10_20.1 [symbolic = %other.patt.loc10_20.2 (constants.%other.patt.c41)]
// CHECK:STDOUT:       %return.param_patt.loc10_31.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = out_param_pattern [symbolic = %return.param_patt.loc10_31.2 (constants.%return.param_patt.8f2)]
// CHECK:STDOUT:       %return.patt.loc10_28.1: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = return_slot_pattern %return.param_patt.loc10_31.1, %.loc10_31.3 [symbolic = %return.patt.loc10_28.2 (constants.%return.patt.46c)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc10_31.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = specific_constant @AddWith.%Self.loc9_28.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:       %Self.ref.loc10_31: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = name_ref Self, %.loc10_31.2 [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:       %Self.as_type.loc10_31: type = facet_access_type %Self.ref.loc10_31 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:       %.loc10_31.3: type = converted %Self.ref.loc10_31, %Self.as_type.loc10_31 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:       %.loc10_31.4: Core.Form = init_form %.loc10_31.3 [symbolic = %.loc10_31.1 (constants.%.581)]
// CHECK:STDOUT:       %self.param: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = value_param call_param0
// CHECK:STDOUT:       %.loc10_9.1: type = splice_block %.loc10_9.3 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)] {
// CHECK:STDOUT:         %.loc10_9.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = specific_constant @AddWith.%Self.loc9_28.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:         %Self.ref.loc10_9: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = name_ref Self, %.loc10_9.2 [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:         %Self.as_type.loc10_9.2: type = facet_access_type %Self.ref.loc10_9 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:         %.loc10_9.3: type = converted %Self.ref.loc10_9, %Self.as_type.loc10_9.2 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = value_param call_param1
// CHECK:STDOUT:       %.loc10_22.1: type = splice_block %.loc10_22.3 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)] {
// CHECK:STDOUT:         %.loc10_22.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = specific_constant @AddWith.%Self.loc9_28.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:         %Self.ref.loc10_22: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = name_ref Self, %.loc10_22.2 [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:         %Self.as_type.loc10_22: type = facet_access_type %Self.ref.loc10_22 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:         %.loc10_22.3: type = converted %Self.ref.loc10_22, %Self.as_type.loc10_22 [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = wrapper_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc10_35.1: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.1d5) = assoc_entity element0, %AddWith.WithSelf.Op.decl [symbolic = %assoc0.loc10_35.2 (constants.%assoc0.728)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc9_28.1
// CHECK:STDOUT:     .Op = @AddWith.WithSelf.%assoc0.loc10_35.1
// CHECK:STDOUT:     witness = (@AddWith.WithSelf.%AddWith.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc13_15.2: type) {
// CHECK:STDOUT:   %T.patt.loc13_15.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_15.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc13_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc13_15.1)> [symbolic = %As.type (constants.%As.type.12c)]
// CHECK:STDOUT:   %Self.loc13_23.2: @As.%As.type (%As.type.12c) = symbolic_binding Self, 1 [symbolic = %Self.loc13_23.2 (constants.%Self.b18)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc13_23.1: @As.%As.type (%As.type.12c) = symbolic_binding Self, 1 [symbolic = %Self.loc13_23.2 (constants.%Self.b18)]
// CHECK:STDOUT:     %As.WithSelf.decl = interface_with_self_decl @As [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %As.WithSelf.Convert.decl: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.40e) = fn_decl @As.WithSelf.Convert [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.d7e)] {
// CHECK:STDOUT:       %self.param_patt.loc14_14.1: @As.WithSelf.Convert.%pattern_type.loc14_14 (%pattern_type.f4a) = value_param_pattern [symbolic = %self.param_patt.loc14_14.2 (constants.%self.param_patt.915)]
// CHECK:STDOUT:       %self.patt.loc14_14.1: @As.WithSelf.Convert.%pattern_type.loc14_14 (%pattern_type.f4a) = wrapper_binding_pattern self, %self.param_patt.loc14_14.1 [symbolic = %self.patt.loc14_14.2 (constants.%self.patt.3aa)]
// CHECK:STDOUT:       %return.param_patt.loc14_23.1: @As.WithSelf.Convert.%pattern_type.loc14_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc14_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc14_20.1: @As.WithSelf.Convert.%pattern_type.loc14_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc14_23.1, %T.ref [symbolic = %return.patt.loc14_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc13_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc14_23.2: Core.Form = init_form %T.ref [symbolic = %.loc14_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @As.WithSelf.Convert.%Self.as_type.loc14_14.1 (%Self.as_type.5c1) = value_param call_param0
// CHECK:STDOUT:       %.loc14_14.1: type = splice_block %.loc14_14.3 [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type.5c1)] {
// CHECK:STDOUT:         %.loc14_14.2: @As.WithSelf.Convert.%As.type (%As.type.12c) = specific_constant @As.%Self.loc13_23.1, @As(constants.%T) [symbolic = %Self (constants.%Self.b18)]
// CHECK:STDOUT:         %Self.ref: @As.WithSelf.Convert.%As.type (%As.type.12c) = name_ref Self, %.loc14_14.2 [symbolic = %Self (constants.%Self.b18)]
// CHECK:STDOUT:         %Self.as_type.loc14_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type.5c1)]
// CHECK:STDOUT:         %.loc14_14.3: type = converted %Self.ref, %Self.as_type.loc14_14.2 [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type.5c1)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.WithSelf.Convert.%Self.as_type.loc14_14.1 (%Self.as_type.5c1) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc14_24.1: @As.WithSelf.%As.assoc_type (%As.assoc_type.e47) = assoc_entity element0, %As.WithSelf.Convert.decl [symbolic = %assoc0.loc14_24.2 (constants.%assoc0.722)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc13_23.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @As.WithSelf.%assoc0.loc14_24.1
// CHECK:STDOUT:     witness = (@As.WithSelf.%As.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc17_23.2: type) {
// CHECK:STDOUT:   %T.patt.loc17_23.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_23.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc17_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc17_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc17_23.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self.loc17_31.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self.loc17_31.2 (constants.%Self.757)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc17_31.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self.loc17_31.2 (constants.%Self.757)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.1b3) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.6e2)] {
// CHECK:STDOUT:       %self.param_patt.loc18_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_14 (%pattern_type.97b) = value_param_pattern [symbolic = %self.param_patt.loc18_14.2 (constants.%self.param_patt.163)]
// CHECK:STDOUT:       %self.patt.loc18_14.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_14 (%pattern_type.97b) = wrapper_binding_pattern self, %self.param_patt.loc18_14.1 [symbolic = %self.patt.loc18_14.2 (constants.%self.patt.da1)]
// CHECK:STDOUT:       %return.param_patt.loc18_23.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc18_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:       %return.patt.loc18_20.1: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc18_23.1, %T.ref [symbolic = %return.patt.loc18_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc17_23.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc18_23.2: Core.Form = init_form %T.ref [symbolic = %.loc18_23.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc18_14.1 (%Self.as_type.0dd) = value_param call_param0
// CHECK:STDOUT:       %.loc18_14.1: type = splice_block %.loc18_14.3 [symbolic = %Self.as_type.loc18_14.1 (constants.%Self.as_type.0dd)] {
// CHECK:STDOUT:         %.loc18_14.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = specific_constant @ImplicitAs.%Self.loc17_31.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.757)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = name_ref Self, %.loc18_14.2 [symbolic = %Self (constants.%Self.757)]
// CHECK:STDOUT:         %Self.as_type.loc18_14.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc18_14.1 (constants.%Self.as_type.0dd)]
// CHECK:STDOUT:         %.loc18_14.3: type = converted %Self.ref, %Self.as_type.loc18_14.2 [symbolic = %Self.as_type.loc18_14.1 (constants.%Self.as_type.0dd)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc18_14.1 (%Self.as_type.0dd) = wrapper_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc18_24.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.07d) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc18_24.2 (constants.%assoc0.0ee)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc17_31.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc18_24.1
// CHECK:STDOUT:     .IntLiteral = <poisoned>
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: %.loc21_6 as %AddWith.type {
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.decl: %i32.builtin.as.AddWith.impl.Op.type = fn_decl @i32.builtin.as.AddWith.impl.Op [concrete = constants.%i32.builtin.as.AddWith.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%self.param_patt.936]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.16e]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%other.param_patt.36d]
// CHECK:STDOUT:     %other.patt: %pattern_type.956 = wrapper_binding_pattern other, %other.param_patt [concrete = constants.%other.patt.8f0]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %Self.ref.loc22_31 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc22_31: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc21_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc22: Core.Form = init_form %Self.ref.loc22_31 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc22_9: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc21_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc22_22: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc21_6 [concrete = constants.%i32.builtin]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d07]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc26_23.1 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc26_23.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc26_23.2: Core.Form = init_form %.loc26_23.1 [concrete = constants.%.9cb]
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
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.d07]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete = constants.%return.param_patt.903]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc30_23.1 [concrete = constants.%return.patt.930]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc30_23.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc30_23.2: Core.Form = init_form %.loc30_23.1 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.ImplicitAs.impl.%IntLiteral.ref [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.c40]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: %.loc33 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.decl: %i32.builtin.as.ImplicitAs.impl.Convert.type = fn_decl @i32.builtin.as.ImplicitAs.impl.Convert [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete = constants.%self.param_patt.936]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.16e]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete = constants.%return.param_patt.eab]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %IntLiteral.ref [concrete = constants.%return.patt.cd4]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: type = name_ref IntLiteral, file.%IntLiteral [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc34: Core.Form = init_form %IntLiteral.ref [concrete = constants.%.f7d]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @i32.builtin.as.ImplicitAs.impl.%.loc33 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = wrapper_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.780]
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
// CHECK:STDOUT: fn @Array(%T.param: type, %N.param: Core.IntLiteral) -> out %return.param: type = "array.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(@AddWith.%T.loc9_20.2: type, @AddWith.%Self.loc9_28.1: @AddWith.%AddWith.type (%AddWith.type.872)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.872)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.fe0)]
// CHECK:STDOUT:   %Self.as_type.loc10_9.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc10_9.1 (constants.%Self.as_type.544)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc10_9.1 [symbolic = %pattern_type (constants.%pattern_type.458)]
// CHECK:STDOUT:   %self.param_patt.loc10_9.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = value_param_pattern [symbolic = %self.param_patt.loc10_9.2 (constants.%self.param_patt.445)]
// CHECK:STDOUT:   %self.patt.loc10_9.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = wrapper_binding_pattern self, %self.param_patt.loc10_9.2 [symbolic = %self.patt.loc10_9.2 (constants.%self.patt.740)]
// CHECK:STDOUT:   %other.param_patt.loc10_20.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = value_param_pattern [symbolic = %other.param_patt.loc10_20.2 (constants.%other.param_patt.1dd)]
// CHECK:STDOUT:   %other.patt.loc10_20.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = wrapper_binding_pattern other, %other.param_patt.loc10_20.2 [symbolic = %other.patt.loc10_20.2 (constants.%other.patt.c41)]
// CHECK:STDOUT:   %.loc10_31.1: Core.Form = init_form %Self.as_type.loc10_9.1 [symbolic = %.loc10_31.1 (constants.%.581)]
// CHECK:STDOUT:   %return.param_patt.loc10_31.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = out_param_pattern [symbolic = %return.param_patt.loc10_31.2 (constants.%return.param_patt.8f2)]
// CHECK:STDOUT:   %return.patt.loc10_28.2: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.458) = return_slot_pattern %return.param_patt.loc10_31.2, %Self.as_type.loc10_9.1 [symbolic = %return.patt.loc10_28.2 (constants.%return.patt.46c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544), %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544)) -> out %return.param: @AddWith.WithSelf.Op.%Self.as_type.loc10_9.1 (%Self.as_type.544);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(@As.%T.loc13_15.2: type, @As.%Self.loc13_23.1: @As.%As.type (%As.type.12c)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.12c)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.12c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.b18)]
// CHECK:STDOUT:   %Self.as_type.loc14_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc14_14.1 (constants.%Self.as_type.5c1)]
// CHECK:STDOUT:   %pattern_type.loc14_14: type = pattern_type %Self.as_type.loc14_14.1 [symbolic = %pattern_type.loc14_14 (constants.%pattern_type.f4a)]
// CHECK:STDOUT:   %self.param_patt.loc14_14.2: @As.WithSelf.Convert.%pattern_type.loc14_14 (%pattern_type.f4a) = value_param_pattern [symbolic = %self.param_patt.loc14_14.2 (constants.%self.param_patt.915)]
// CHECK:STDOUT:   %self.patt.loc14_14.2: @As.WithSelf.Convert.%pattern_type.loc14_14 (%pattern_type.f4a) = wrapper_binding_pattern self, %self.param_patt.loc14_14.2 [symbolic = %self.patt.loc14_14.2 (constants.%self.patt.3aa)]
// CHECK:STDOUT:   %.loc14_23.1: Core.Form = init_form %T [symbolic = %.loc14_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc14_23: type = pattern_type %T [symbolic = %pattern_type.loc14_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc14_23.2: @As.WithSelf.Convert.%pattern_type.loc14_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc14_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc14_20.2: @As.WithSelf.Convert.%pattern_type.loc14_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc14_23.2, %T [symbolic = %return.patt.loc14_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.WithSelf.Convert.%Self.as_type.loc14_14.1 (%Self.as_type.5c1)) -> out %return.param: @As.WithSelf.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%T.loc17_23.2: type, @ImplicitAs.%Self.loc17_31.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.757)]
// CHECK:STDOUT:   %Self.as_type.loc18_14.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc18_14.1 (constants.%Self.as_type.0dd)]
// CHECK:STDOUT:   %pattern_type.loc18_14: type = pattern_type %Self.as_type.loc18_14.1 [symbolic = %pattern_type.loc18_14 (constants.%pattern_type.97b)]
// CHECK:STDOUT:   %self.param_patt.loc18_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_14 (%pattern_type.97b) = value_param_pattern [symbolic = %self.param_patt.loc18_14.2 (constants.%self.param_patt.163)]
// CHECK:STDOUT:   %self.patt.loc18_14.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_14 (%pattern_type.97b) = wrapper_binding_pattern self, %self.param_patt.loc18_14.2 [symbolic = %self.patt.loc18_14.2 (constants.%self.patt.da1)]
// CHECK:STDOUT:   %.loc18_23.1: Core.Form = init_form %T [symbolic = %.loc18_23.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc18_23: type = pattern_type %T [symbolic = %pattern_type.loc18_23 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc18_23.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_23 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc18_23.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc18_20.2: @ImplicitAs.WithSelf.Convert.%pattern_type.loc18_23 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc18_23.2, %T [symbolic = %return.patt.loc18_20.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc18_14.1 (%Self.as_type.0dd)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%T (%T);
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
// CHECK:STDOUT:   %T.patt.loc9_20.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc9_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.fe0) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.fe0) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.872
// CHECK:STDOUT:   %Self => constants.%Self.fe0
// CHECK:STDOUT:   %Self.as_type.loc10_9.1 => constants.%Self.as_type.544
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.458
// CHECK:STDOUT:   %self.param_patt.loc10_9.2 => constants.%self.param_patt.445
// CHECK:STDOUT:   %self.patt.loc10_9.2 => constants.%self.patt.740
// CHECK:STDOUT:   %other.param_patt.loc10_20.2 => constants.%other.param_patt.1dd
// CHECK:STDOUT:   %other.patt.loc10_20.2 => constants.%other.patt.c41
// CHECK:STDOUT:   %.loc10_31.1 => constants.%.581
// CHECK:STDOUT:   %return.param_patt.loc10_31.2 => constants.%return.param_patt.8f2
// CHECK:STDOUT:   %return.patt.loc10_28.2 => constants.%return.patt.46c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.b18) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.b18) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.12c
// CHECK:STDOUT:   %Self => constants.%Self.b18
// CHECK:STDOUT:   %Self.as_type.loc14_14.1 => constants.%Self.as_type.5c1
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.f4a
// CHECK:STDOUT:   %self.param_patt.loc14_14.2 => constants.%self.param_patt.915
// CHECK:STDOUT:   %self.patt.loc14_14.2 => constants.%self.patt.3aa
// CHECK:STDOUT:   %.loc14_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc14_23 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc14_23.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc14_20.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc17_23.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc17_23.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.757) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.757) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.b02
// CHECK:STDOUT:   %Self => constants.%Self.757
// CHECK:STDOUT:   %Self.as_type.loc18_14.1 => constants.%Self.as_type.0dd
// CHECK:STDOUT:   %pattern_type.loc18_14 => constants.%pattern_type.97b
// CHECK:STDOUT:   %self.param_patt.loc18_14.2 => constants.%self.param_patt.163
// CHECK:STDOUT:   %self.patt.loc18_14.2 => constants.%self.patt.da1
// CHECK:STDOUT:   %.loc18_23.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc18_23 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc18_23.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc18_20.2 => constants.%return.patt.b39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc9_20.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc9_20.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self.loc9_28.2 => constants.%Self.69c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.fe0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%Self.fe0
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.139
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.9b0
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ec
// CHECK:STDOUT:   %assoc0.loc10_35.2 => constants.%assoc0.33e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.872
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.6e9
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ec
// CHECK:STDOUT:   %assoc0.loc10_35.2 => constants.%assoc0.33e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.as_type.loc10_9.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT:   %self.param_patt.loc10_9.2 => constants.%self.param_patt.936
// CHECK:STDOUT:   %self.patt.loc10_9.2 => constants.%self.patt.16e
// CHECK:STDOUT:   %other.param_patt.loc10_20.2 => constants.%other.param_patt.36d
// CHECK:STDOUT:   %other.patt.loc10_20.2 => constants.%other.patt.8f0
// CHECK:STDOUT:   %.loc10_31.1 => constants.%.9cb
// CHECK:STDOUT:   %return.param_patt.loc10_31.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc10_28.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc13_15.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc13_15.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self.loc13_23.2 => constants.%Self.1d5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.b18) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%Self.b18
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.046
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.e01
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.12b
// CHECK:STDOUT:   %assoc0.loc14_24.2 => constants.%assoc0.092
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.c8a
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.baa
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.12b
// CHECK:STDOUT:   %assoc0.loc14_24.2 => constants.%assoc0.092
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc14_14.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc14_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %self.param_patt.loc14_14.2 => constants.%self.param_patt.1ca
// CHECK:STDOUT:   %self.patt.loc14_14.2 => constants.%self.patt.d07
// CHECK:STDOUT:   %.loc14_23.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc14_23 => constants.%pattern_type.956
// CHECK:STDOUT:   %return.param_patt.loc14_23.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc14_20.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt.loc17_23.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc17_23.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self.loc17_31.2 => constants.%Self.0be
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.757) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%Self.757
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.6f6
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.4a7
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.d73
// CHECK:STDOUT:   %assoc0.loc18_24.2 => constants.%assoc0.a2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.1e5) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.1e5
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.c1c
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.a8a
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.d73
// CHECK:STDOUT:   %assoc0.loc18_24.2 => constants.%assoc0.a2d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.1e5) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.1e5
// CHECK:STDOUT:   %Self.as_type.loc18_14.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc18_14 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %self.param_patt.loc18_14.2 => constants.%self.param_patt.1ca
// CHECK:STDOUT:   %self.patt.loc18_14.2 => constants.%self.patt.d07
// CHECK:STDOUT:   %.loc18_23.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc18_23 => constants.%pattern_type.956
// CHECK:STDOUT:   %return.param_patt.loc18_23.2 => constants.%return.param_patt.903
// CHECK:STDOUT:   %return.patt.loc18_20.2 => constants.%return.patt.930
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.patt.loc17_23.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc17_23.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self.loc17_31.2 => constants.%Self.ee1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.757) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%Self.757
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.0bc
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.67f
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.32b
// CHECK:STDOUT:   %assoc0.loc18_24.2 => constants.%assoc0.879
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.69f) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.69f
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.c18
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.d73
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.32b
// CHECK:STDOUT:   %assoc0.loc18_24.2 => constants.%assoc0.879
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.69f) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.69f
// CHECK:STDOUT:   %Self.as_type.loc18_14.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type.loc18_14 => constants.%pattern_type.956
// CHECK:STDOUT:   %self.param_patt.loc18_14.2 => constants.%self.param_patt.936
// CHECK:STDOUT:   %self.patt.loc18_14.2 => constants.%self.patt.16e
// CHECK:STDOUT:   %.loc18_23.1 => constants.%.f7d
// CHECK:STDOUT:   %pattern_type.loc18_23 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %return.param_patt.loc18_23.2 => constants.%return.param_patt.eab
// CHECK:STDOUT:   %return.patt.loc18_20.2 => constants.%return.patt.cd4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.116: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.116 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.12c: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Self.03d: %As.type.12c = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.e47: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.b5f: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.03d) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.3da: %As.WithSelf.Convert.type.b5f = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.c39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, invalid [symbolic]
// CHECK:STDOUT:   %Self.as_type.ca4: type = facet_access_type %Self.03d [symbolic]
// CHECK:STDOUT:   %pattern_type.19c: type = pattern_type %Self.as_type.ca4 [symbolic]
// CHECK:STDOUT:   %self.param_patt.573: %pattern_type.19c = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.788: %pattern_type.19c = wrapper_binding_pattern self, %self.param_patt.573 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %As.type.294: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.9c1: %As.type.294 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.383: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.03d) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.fa8: %As.WithSelf.Convert.type.383 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.12b: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.c37: %As.assoc_type.12b = assoc_entity element0, imports.%Core.import_ref.d19 [concrete]
// CHECK:STDOUT:   %assoc0.57c: %As.assoc_type.e47 = assoc_entity element0, imports.%Core.import_ref.40e [symbolic]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.294 = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.773: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.116: %As.WithSelf.Convert.type.773 = struct_value () [concrete]
// CHECK:STDOUT:   %.329: type = fn_type_with_self_type %As.WithSelf.Convert.type.773, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.5d7: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.959: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.2ef: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.2ef = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.872: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.958: %AddWith.type.872 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.1d5: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.70b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.bb1: %AddWith.WithSelf.Op.type.70b = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.0b7: type = facet_access_type %Self.958 [symbolic]
// CHECK:STDOUT:   %pattern_type.be7: type = pattern_type %Self.as_type.0b7 [symbolic]
// CHECK:STDOUT:   %return.param_patt.430: %pattern_type.be7 = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.824: %pattern_type.be7 = return_slot_pattern %return.param_patt.430, invalid [symbolic]
// CHECK:STDOUT:   %self.param_patt.286: %pattern_type.be7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.271: %pattern_type.be7 = wrapper_binding_pattern self, %self.param_patt.286 [symbolic]
// CHECK:STDOUT:   %.c2d: Core.Form = init_form %Self.as_type.0b7 [symbolic]
// CHECK:STDOUT:   %other.param_patt.218: %pattern_type.be7 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %other.patt.48d: %pattern_type.be7 = wrapper_binding_pattern other, %other.param_patt.218 [symbolic]
// CHECK:STDOUT:   %AddWith.type.8a5: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.a35: %AddWith.type.8a5 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.93f: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.958) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.80e: %AddWith.WithSelf.Op.type.93f = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.5ec: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.c91: %AddWith.assoc_type.5ec = assoc_entity element0, imports.%Core.import_ref.8ad [concrete]
// CHECK:STDOUT:   %assoc0.6eb: %AddWith.assoc_type.1d5 = assoc_entity element0, imports.%Core.import_ref.e9a [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.8a5 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.78b: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.879: %AddWith.WithSelf.Op.type.78b = struct_value () [concrete]
// CHECK:STDOUT:   %.643: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.78b, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.351: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b02: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.862: %ImplicitAs.type.b02 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.07d: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.eae: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.862) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.699: %ImplicitAs.WithSelf.Convert.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.ef4: type = facet_access_type %Self.862 [symbolic]
// CHECK:STDOUT:   %pattern_type.662: type = pattern_type %Self.as_type.ef4 [symbolic]
// CHECK:STDOUT:   %self.param_patt.f94: %pattern_type.662 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %self.patt.97f: %pattern_type.662 = wrapper_binding_pattern self, %self.param_patt.f94 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.5be: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.47b: %ImplicitAs.type.5be = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.40d: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.862) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.d45: %ImplicitAs.WithSelf.Convert.type.40d = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.32b: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.994: %ImplicitAs.assoc_type.32b = assoc_entity element0, imports.%Core.import_ref.a03 [concrete]
// CHECK:STDOUT:   %assoc0.3a5: %ImplicitAs.assoc_type.07d = assoc_entity element0, imports.%Core.import_ref.9e1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.cb4: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.306: %ImplicitAs.type.cb4 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.5a8: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.34b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.aa0: %ImplicitAs.type.5be = facet_value %i32.builtin, (%ImplicitAs.impl_witness.5a8) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.54b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.aa0) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.414: %ImplicitAs.WithSelf.Convert.type.54b = struct_value () [concrete]
// CHECK:STDOUT:   %.189: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.54b, %ImplicitAs.facet.aa0 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %Array.type: type = fn_type @Array [concrete]
// CHECK:STDOUT:   %Array: %Array.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.305: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.862) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.e61: %ImplicitAs.WithSelf.Convert.type.305 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.d73: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.158: %ImplicitAs.assoc_type.d73 = assoc_entity element0, imports.%Core.import_ref.a03 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e70: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1e0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.a1b: %ImplicitAs.type.cb4 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e70) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.383: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.a1b) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.029: %ImplicitAs.WithSelf.Convert.type.383 = struct_value () [concrete]
// CHECK:STDOUT:   %.b42: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.383, %ImplicitAs.facet.a1b [concrete]
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
// CHECK:STDOUT:     .Array = %Core.Array
// CHECK:STDOUT:     import Core//default
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//default, Int, loaded [concrete = constants.%Int]
// CHECK:STDOUT:   %Core.As: %As.type.116 = import_ref Core//default, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.efc: @As.WithSelf.%As.assoc_type (%As.assoc_type.e47) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%assoc0 (constants.%assoc0.57c)]
// CHECK:STDOUT:   %Core.Convert.ec1 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.d19: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.b5f) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.3da)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.f5cc7e.2: @As.%As.type (%As.type.12c) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%Self (constants.%Self.03d)]
// CHECK:STDOUT:   %Core.import_ref.4be = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.40e = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b95: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.2dc10a.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.ba7: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.type.294]
// CHECK:STDOUT:   %Core.import_ref.06c923.1: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.06c923.1), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.2ef = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.1f2: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.1d5) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc0 (constants.%assoc0.6eb)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.8ad: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.70b) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.bb1)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.5: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.72d00e.2: @AddWith.%AddWith.type (%AddWith.type.872) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%Self (constants.%Self.958)]
// CHECK:STDOUT:   %Core.import_ref.41a = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.e9a = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.539: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.883: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.type.8a5]
// CHECK:STDOUT:   %Core.import_ref.9c3: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.383: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.07d) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.3a5)]
// CHECK:STDOUT:   %Core.Convert.f1f = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.a03: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.eae) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.699)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.8: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.5e2fa1.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.862)]
// CHECK:STDOUT:   %Core.import_ref.3a2 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.9: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.9e1 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.f53: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.e70]
// CHECK:STDOUT:   %Core.import_ref.2dc10a.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.563: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.cb4]
// CHECK:STDOUT:   %Core.import_ref.d3c: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.5a8]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.f4b: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.5be]
// CHECK:STDOUT:   %Core.import_ref.06c923.2: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.34b = impl_witness_table (%Core.import_ref.06c923.2), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Array: %Array.type = import_ref Core//default, Array, loaded [concrete = constants.%Array]
// CHECK:STDOUT:   %Core.import_ref.06c923.3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1e0 = impl_witness_table (%Core.import_ref.06c923.3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %arr.var: ref %array_type = var_storage %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_44.1: type = splice_block %.loc4_44.3 [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc4_16: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc4_27: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.329 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.5d7]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc4_40: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.329 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.959]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.643 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.351]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.189 = impl_witness_access constants.%ImplicitAs.impl_witness.5a8, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %i32.builtin.as.AddWith.impl.Op.call, %impl.elem0.loc4_32.2 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %i32.builtin.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %i32.builtin.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %Array.call: init type = call constants.%Array(%.loc4_16, %.loc4_32.4) [concrete = constants.%array_type]
// CHECK:STDOUT:     %.loc4_44.2: type = value_of_initializer %Array.call [concrete = constants.%array_type]
// CHECK:STDOUT:     %.loc4_44.3: type = converted %Array.call, %.loc4_44.2 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = wrapper_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.9e2 = ref_binding_pattern arr [concrete = constants.%arr.patt]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.9e2 = var_pattern %arr.patt [concrete = constants.%arr.var_patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.12c)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.12c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.03d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.4be
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.efc
// CHECK:STDOUT:     witness = (imports.%Core.Convert.ec1)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.b3bc94.6: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.872)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.872) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.958)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.41a
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.1f2
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.9: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.862)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.3a2
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.383
// CHECK:STDOUT:     witness = (imports.%Core.Convert.f1f)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.2dc10a.1 as imports.%Core.import_ref.ba7 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.b95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.63cbb1.2 as imports.%Core.import_ref.883 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.539
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.2dc10a.2 as imports.%Core.import_ref.563 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.f53
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.63cbb1.3 as imports.%Core.import_ref.f4b [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.d3c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.f5cc7e.2: @As.%As.type (%As.type.12c)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.12c)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.12c) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.03d)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.ca4)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.19c)]
// CHECK:STDOUT:   %self.param_patt: @As.WithSelf.Convert.%pattern_type.1 (%pattern_type.19c) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.573)]
// CHECK:STDOUT:   %self.patt: @As.WithSelf.Convert.%pattern_type.1 (%pattern_type.19c) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.788)]
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
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(imports.%Core.import_ref.b3bc94.5: type, imports.%Core.import_ref.72d00e.2: @AddWith.%AddWith.type (%AddWith.type.872)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.872)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.872) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.958)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.0b7)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.be7)]
// CHECK:STDOUT:   %self.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.286)]
// CHECK:STDOUT:   %self.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.271)]
// CHECK:STDOUT:   %other.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = value_param_pattern [symbolic = %other.param_patt (constants.%other.param_patt.218)]
// CHECK:STDOUT:   %other.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = wrapper_binding_pattern other, %other.param_patt [symbolic = %other.patt (constants.%other.patt.48d)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Self.as_type [symbolic = %.1 (constants.%.c2d)]
// CHECK:STDOUT:   %return.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = out_param_pattern [symbolic = %return.param_patt (constants.%return.param_patt.430)]
// CHECK:STDOUT:   %return.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.be7) = return_slot_pattern %return.param_patt, invalid [symbolic = %return.patt (constants.%return.patt.824)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.8: type, imports.%Core.import_ref.5e2fa1.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.b02)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.b02)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.b02) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.862)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.ef4)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.662)]
// CHECK:STDOUT:   %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.1 (%pattern_type.662) = value_param_pattern [symbolic = %self.param_patt (constants.%self.param_patt.f94)]
// CHECK:STDOUT:   %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.1 (%pattern_type.662) = wrapper_binding_pattern self, %self.param_patt [symbolic = %self.patt (constants.%self.patt.97f)]
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
// CHECK:STDOUT: fn @Array = "array.make_type" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.ImplicitAs.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %int_3.loc4_49: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %int_4.loc4_52: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %int_3.loc4_56: Core.IntLiteral = int_value 3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:   %.loc4_61: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.329 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.94b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc4_74: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.329 = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.0f9]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.643 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.c75]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.b42 = impl_witness_access constants.%ImplicitAs.impl_witness.e70, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.94b]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin to %.loc4_78.3 = in_place_init %.loc4_78.2 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.b42 = impl_witness_access constants.%ImplicitAs.impl_witness.e70, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
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
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.03d) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.03d) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.12c
// CHECK:STDOUT:   %Self => constants.%Self.03d
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.ca4
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.19c
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.573
// CHECK:STDOUT:   %self.patt => constants.%self.patt.788
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%Self.9c1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.03d) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%Self.03d
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.383
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.fa8
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.12b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.294
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.773
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.116
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.12b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.958) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.958) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.872
// CHECK:STDOUT:   %Self => constants.%Self.958
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.0b7
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.be7
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.286
// CHECK:STDOUT:   %self.patt => constants.%self.patt.271
// CHECK:STDOUT:   %other.param_patt => constants.%other.param_patt.218
// CHECK:STDOUT:   %other.patt => constants.%other.patt.48d
// CHECK:STDOUT:   %.1 => constants.%.c2d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.430
// CHECK:STDOUT:   %return.patt => constants.%return.patt.824
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%Self.a35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.958) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%Self.958
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.93f
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.80e
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ec
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.8a5
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.78b
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.879
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.5ec
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.c91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.862) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.862) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.b02
// CHECK:STDOUT:   %Self => constants.%Self.862
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.ef4
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.662
// CHECK:STDOUT:   %self.param_patt => constants.%self.param_patt.f94
// CHECK:STDOUT:   %self.patt => constants.%self.patt.97f
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt => constants.%return.patt.c39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%Self.47b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.862) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%Self.862
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.40d
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.d45
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.32b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.994
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.patt => constants.%T.patt.3a0
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%Self.306
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.aa0) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.5be
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.aa0
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.54b
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.414
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.32b
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.994
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.862) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%Self.862
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.305
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.e61
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.d73
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.a1b) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.cb4
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.a1b
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.383
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.029
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.d73
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.158
// CHECK:STDOUT: }
// CHECK:STDOUT:
