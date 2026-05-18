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

fn IntLiteral() -> type = "int_literal.make_type";
fn Int(N: IntLiteral()) -> type = "int.make_type_signed";

interface AddWith(T:! type) {
  fn Op[self: Self](other: Self) -> Self;
}

interface As(T:! type) {
  fn Convert[self: Self]() -> T;
}

interface ImplicitAs(T:! type) {
  fn Convert[self: Self]() -> T;
}

impl i32 as AddWith(i32) {
  fn Op[self: Self](other: Self) -> Self = "int.sadd";
}

impl IntLiteral() as As(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl IntLiteral() as ImplicitAs(i32) {
  fn Convert[self: Self]() -> i32 = "int.convert_checked";
}

impl i32 as ImplicitAs(IntLiteral()) {
  fn Convert[self: Self]() -> IntLiteral() = "int.convert_checked";
}

// --- user.carbon

import Core;

var arr: array(i32, (1 as i32) + (2 as i32)) = (3, 4, (3 as i32) + (4 as i32));

// CHECK:STDOUT: --- core.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.805: Core.Form = init_form type [concrete]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %IntLiteral.type: type = fn_type @IntLiteral [concrete]
// CHECK:STDOUT:   %IntLiteral: %IntLiteral.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.dc0: type = pattern_type Core.IntLiteral [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %AddWith.type.5e0: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.5e0 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.adb: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.1fa: %AddWith.type.adb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.9bf: type = facet_access_type %Self.1fa [symbolic]
// CHECK:STDOUT:   %pattern_type.ad8: type = pattern_type %Self.as_type.9bf [symbolic]
// CHECK:STDOUT:   %.06b: Core.Form = init_form %Self.as_type.9bf [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.946: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.1fa) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.111: %AddWith.WithSelf.Op.type.946 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.329: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.4d9: %AddWith.assoc_type.329 = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [symbolic]
// CHECK:STDOUT:   %As.type.962: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.962 = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.5df: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.ddb: %As.type.5df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.784: type = facet_access_type %Self.ddb [symbolic]
// CHECK:STDOUT:   %pattern_type.cb5: type = pattern_type %Self.as_type.784 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f9a: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.ddb) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.c43: %As.WithSelf.Convert.type.f9a = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.9d6: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.192: %As.assoc_type.9d6 = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.152: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f53: %ImplicitAs.type.152 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %Self.as_type.423: type = facet_access_type %Self.f53 [symbolic]
// CHECK:STDOUT:   %pattern_type.524: type = pattern_type %Self.as_type.423 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.75f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.f53) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.83a: %ImplicitAs.WithSelf.Convert.type.75f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.6dd: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %assoc0.2fe: %ImplicitAs.assoc_type.6dd = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %AddWith.type.248: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness @i32.builtin.as.AddWith.impl.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.52c: %AddWith.type.248 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.6e5: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.1fa) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.f8a: %AddWith.WithSelf.Op.type.6e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.b08: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.a97: %AddWith.assoc_type.b08 = assoc_entity element0, @AddWith.WithSelf.%AddWith.WithSelf.Op.decl [concrete]
// CHECK:STDOUT:   %pattern_type.956: type = pattern_type %i32.builtin [concrete]
// CHECK:STDOUT:   %.9cb: Core.Form = init_form %i32.builtin [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.248 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.6cc: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.8ae: %AddWith.WithSelf.Op.type.6cc = struct_value () [concrete]
// CHECK:STDOUT:   %As.type.3aa: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness @Core.IntLiteral.as.As.impl.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.2ec: %As.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.ad6: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.ddb) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.e6e: %As.WithSelf.Convert.type.ad6 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.0fc: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.1bf: %As.assoc_type.0fc = assoc_entity element0, @As.WithSelf.%As.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.3aa = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.dfe: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.d50: %As.WithSelf.Convert.type.dfe = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.c36: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e40: <witness> = impl_witness @Core.IntLiteral.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.061: %ImplicitAs.type.c36 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.7cb: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.f53) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.ae1: %ImplicitAs.WithSelf.Convert.type.7cb = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.aa1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.715: %ImplicitAs.assoc_type.aa1 = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.b41: %ImplicitAs.type.c36 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e40) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.2a8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.b41) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.b67: %ImplicitAs.WithSelf.Convert.type.2a8 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.ab8: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.57e: <witness> = impl_witness @i32.builtin.as.ImplicitAs.impl.%ImplicitAs.impl_witness_table [concrete]
// CHECK:STDOUT:   %Self.98d: %ImplicitAs.type.ab8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.801: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.f53) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.4dc: %ImplicitAs.WithSelf.Convert.type.801 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.49c: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.549: %ImplicitAs.assoc_type.49c = assoc_entity element0, @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl [concrete]
// CHECK:STDOUT:   %.f7d: Core.Form = init_form Core.IntLiteral [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.529: %ImplicitAs.type.ab8 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.57e) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.3ed: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.529) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.d26: %ImplicitAs.WithSelf.Convert.type.3ed = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .IntLiteral = %IntLiteral.decl
// CHECK:STDOUT:     .Int = %Int.decl
// CHECK:STDOUT:     .AddWith = %AddWith.decl
// CHECK:STDOUT:     .As = %As.decl
// CHECK:STDOUT:     .ImplicitAs = %ImplicitAs.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %IntLiteral.decl: %IntLiteral.type = fn_decl @IntLiteral [concrete = constants.%IntLiteral] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc4_20.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_20.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc4_20.2: Core.Form = init_form %.loc4_20.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %return.param: ref type = out_param call_param0
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Int.decl: %Int.type = fn_decl @Int [concrete = constants.%Int] {
// CHECK:STDOUT:     %N.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %N.patt: %pattern_type.dc0 = at_binding_pattern N, %N.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98f = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.98f = return_slot_pattern %return.param_patt, %.loc5_28.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc5_28.1: type = type_literal type [concrete = type]
// CHECK:STDOUT:     %.loc5_28.2: Core.Form = init_form %.loc5_28.1 [concrete = constants.%.805]
// CHECK:STDOUT:     %N.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %.loc5_22.1: type = splice_block %.loc5_22.3 [concrete = Core.IntLiteral] {
// CHECK:STDOUT:       %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:       %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.2: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:       %.loc5_22.3: type = converted %IntLiteral.call, %.loc5_22.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N: Core.IntLiteral = value_binding N, %N.param
// CHECK:STDOUT:     %return.param: ref type = out_param call_param1
// CHECK:STDOUT:     %return: ref type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %AddWith.decl: %AddWith.type.5e0 = interface_decl @AddWith [concrete = constants.%AddWith.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_23.1: type = splice_block %.loc7_23.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_23.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_20.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_20.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %As.decl: %As.type.962 = interface_decl @As [concrete = constants.%As.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc11_18.1: type = splice_block %.loc11_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc11_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc11_15.2: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.decl: %ImplicitAs.type.b23 = interface_decl @ImplicitAs [concrete = constants.%ImplicitAs.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc15_26.1: type = splice_block %.loc15_26.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc15_26.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_23.2: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.AddWith.impl [concrete] {} {
// CHECK:STDOUT:     %.loc19_6: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.ref: %AddWith.type.5e0 = name_ref AddWith, file.%AddWith.decl [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:     %.loc19_21: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %AddWith.type: type = facet_type <@AddWith, @AddWith(constants.%i32.builtin)> [concrete = constants.%AddWith.type.248]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.As.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc23_17.2: type = converted %IntLiteral.call, %.loc23_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %As.ref: %As.type.962 = name_ref As, file.%As.decl [concrete = constants.%As.generic]
// CHECK:STDOUT:     %.loc23_25: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %As.type: type = facet_type <@As, @As(constants.%i32.builtin)> [concrete = constants.%As.type.3aa]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @Core.IntLiteral.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc27_17.2: type = converted %IntLiteral.call, %.loc27_17.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %.loc27_33: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(constants.%i32.builtin)> [concrete = constants.%ImplicitAs.type.c36]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   impl_decl @i32.builtin.as.ImplicitAs.impl [concrete] {} {
// CHECK:STDOUT:     %.loc31_6: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %ImplicitAs.ref: %ImplicitAs.type.b23 = name_ref ImplicitAs, file.%ImplicitAs.decl [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc31_36.2: type = converted %IntLiteral.call, %.loc31_36.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete = constants.%ImplicitAs.type.ab8]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(%T.loc7_20.2: type) {
// CHECK:STDOUT:   %T.loc7_20.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_20.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T.loc7_20.1)> [symbolic = %AddWith.type (constants.%AddWith.type.adb)]
// CHECK:STDOUT:   %Self.loc7_29.2: @AddWith.%AddWith.type (%AddWith.type.adb) = symbolic_binding Self, 1 [symbolic = %Self.loc7_29.2 (constants.%Self.1fa)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc7_29.1: @AddWith.%AddWith.type (%AddWith.type.adb) = symbolic_binding Self, 1 [symbolic = %Self.loc7_29.2 (constants.%Self.1fa)]
// CHECK:STDOUT:     %AddWith.WithSelf.decl = interface_with_self_decl @AddWith [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %AddWith.WithSelf.Op.decl: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.946) = fn_decl @AddWith.WithSelf.Op [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.111)] {
// CHECK:STDOUT:       %self.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %other.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = value_param_pattern [concrete]
// CHECK:STDOUT:       %other.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @AddWith.WithSelf.Op.%pattern_type (%pattern_type.ad8) = return_slot_pattern %return.param_patt, %.loc8_37.3 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc8_37.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:       %Self.ref.loc8_37: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = name_ref Self, %.loc8_37.2 [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:       %Self.as_type.loc8_37: type = facet_access_type %Self.ref.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:       %.loc8_37.3: type = converted %Self.ref.loc8_37, %Self.as_type.loc8_37 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:       %.loc8_37.4: Core.Form = init_form %.loc8_37.3 [symbolic = %.loc8_37.1 (constants.%.06b)]
// CHECK:STDOUT:       %self.param: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = value_param call_param0
// CHECK:STDOUT:       %.loc8_15.1: type = splice_block %.loc8_15.3 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)] {
// CHECK:STDOUT:         %.loc8_15.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:         %Self.ref.loc8_15: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = name_ref Self, %.loc8_15.2 [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:         %Self.as_type.loc8_15.2: type = facet_access_type %Self.ref.loc8_15 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:         %.loc8_15.3: type = converted %Self.ref.loc8_15, %Self.as_type.loc8_15.2 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = value_binding self, %self.param
// CHECK:STDOUT:       %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = value_param call_param1
// CHECK:STDOUT:       %.loc8_28.1: type = splice_block %.loc8_28.3 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)] {
// CHECK:STDOUT:         %.loc8_28.2: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = specific_constant @AddWith.%Self.loc7_29.1, @AddWith(constants.%T) [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:         %Self.ref.loc8_28: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = name_ref Self, %.loc8_28.2 [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:         %Self.as_type.loc8_28: type = facet_access_type %Self.ref.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:         %.loc8_28.3: type = converted %Self.ref.loc8_28, %Self.as_type.loc8_28 [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %other: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = value_binding other, %other.param
// CHECK:STDOUT:       %return.param: ref @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = out_param call_param2
// CHECK:STDOUT:       %return: ref @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc8_41.1: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.329) = assoc_entity element0, %AddWith.WithSelf.Op.decl [symbolic = %assoc0.loc8_41.2 (constants.%assoc0.4d9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc7_29.1
// CHECK:STDOUT:     .Op = @AddWith.WithSelf.%assoc0.loc8_41.1
// CHECK:STDOUT:     witness = (@AddWith.WithSelf.%AddWith.WithSelf.Op.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(%T.loc11_15.2: type) {
// CHECK:STDOUT:   %T.loc11_15.1: type = symbolic_binding T, 0 [symbolic = %T.loc11_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T.loc11_15.1)> [symbolic = %As.type (constants.%As.type.5df)]
// CHECK:STDOUT:   %Self.loc11_24.2: @As.%As.type (%As.type.5df) = symbolic_binding Self, 1 [symbolic = %Self.loc11_24.2 (constants.%Self.ddb)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc11_24.1: @As.%As.type (%As.type.5df) = symbolic_binding Self, 1 [symbolic = %Self.loc11_24.2 (constants.%Self.ddb)]
// CHECK:STDOUT:     %As.WithSelf.decl = interface_with_self_decl @As [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %As.WithSelf.Convert.decl: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.f9a) = fn_decl @As.WithSelf.Convert [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.c43)] {
// CHECK:STDOUT:       %self.param_patt: @As.WithSelf.Convert.%pattern_type.loc12_18 (%pattern_type.cb5) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @As.WithSelf.Convert.%pattern_type.loc12_18 (%pattern_type.cb5) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @As.WithSelf.Convert.%pattern_type.loc12_31 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @As.WithSelf.Convert.%pattern_type.loc12_31 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @As.%T.loc11_15.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc12_31.2: Core.Form = init_form %T.ref [symbolic = %.loc12_31.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @As.WithSelf.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.784) = value_param call_param0
// CHECK:STDOUT:       %.loc12_20.1: type = splice_block %.loc12_20.3 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.784)] {
// CHECK:STDOUT:         %.loc12_20.2: @As.WithSelf.Convert.%As.type (%As.type.5df) = specific_constant @As.%Self.loc11_24.1, @As(constants.%T) [symbolic = %Self (constants.%Self.ddb)]
// CHECK:STDOUT:         %Self.ref: @As.WithSelf.Convert.%As.type (%As.type.5df) = name_ref Self, %.loc12_20.2 [symbolic = %Self (constants.%Self.ddb)]
// CHECK:STDOUT:         %Self.as_type.loc12_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.784)]
// CHECK:STDOUT:         %.loc12_20.3: type = converted %Self.ref, %Self.as_type.loc12_20.2 [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.784)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @As.WithSelf.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.784) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @As.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @As.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc12_32.1: @As.WithSelf.%As.assoc_type (%As.assoc_type.9d6) = assoc_entity element0, %As.WithSelf.Convert.decl [symbolic = %assoc0.loc12_32.2 (constants.%assoc0.192)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc11_24.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @As.WithSelf.%assoc0.loc12_32.1
// CHECK:STDOUT:     witness = (@As.WithSelf.%As.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(%T.loc15_23.2: type) {
// CHECK:STDOUT:   %T.loc15_23.1: type = symbolic_binding T, 0 [symbolic = %T.loc15_23.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T.loc15_23.1)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self.loc15_32.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self.loc15_32.2 (constants.%Self.f53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:     %Self.loc15_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self.loc15_32.2 (constants.%Self.f53)]
// CHECK:STDOUT:     %ImplicitAs.WithSelf.decl = interface_with_self_decl @ImplicitAs [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT:   !with Self:
// CHECK:STDOUT:     %ImplicitAs.WithSelf.Convert.decl: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.75f) = fn_decl @ImplicitAs.WithSelf.Convert [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.83a)] {
// CHECK:STDOUT:       %self.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc16_18 (%pattern_type.524) = value_param_pattern [concrete]
// CHECK:STDOUT:       %self.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc16_18 (%pattern_type.524) = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:       %return.param_patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc16_31 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:       %return.patt: @ImplicitAs.WithSelf.Convert.%pattern_type.loc16_31 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @ImplicitAs.%T.loc15_23.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %.loc16_31.2: Core.Form = init_form %T.ref [symbolic = %.loc16_31.1 (constants.%.184)]
// CHECK:STDOUT:       %self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.423) = value_param call_param0
// CHECK:STDOUT:       %.loc16_20.1: type = splice_block %.loc16_20.3 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.423)] {
// CHECK:STDOUT:         %.loc16_20.2: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = specific_constant @ImplicitAs.%Self.loc15_32.1, @ImplicitAs(constants.%T) [symbolic = %Self (constants.%Self.f53)]
// CHECK:STDOUT:         %Self.ref: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = name_ref Self, %.loc16_20.2 [symbolic = %Self (constants.%Self.f53)]
// CHECK:STDOUT:         %Self.as_type.loc16_20.2: type = facet_access_type %Self.ref [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.423)]
// CHECK:STDOUT:         %.loc16_20.3: type = converted %Self.ref, %Self.as_type.loc16_20.2 [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.423)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.423) = value_binding self, %self.param
// CHECK:STDOUT:       %return.param: ref @ImplicitAs.WithSelf.Convert.%T (%T) = out_param call_param1
// CHECK:STDOUT:       %return: ref @ImplicitAs.WithSelf.Convert.%T (%T) = return_slot %return.param
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %assoc0.loc16_32.1: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.6dd) = assoc_entity element0, %ImplicitAs.WithSelf.Convert.decl [symbolic = %assoc0.loc16_32.2 (constants.%assoc0.2fe)]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = %Self.loc15_32.1
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .Convert = @ImplicitAs.WithSelf.%assoc0.loc16_32.1
// CHECK:STDOUT:     .IntLiteral = <poisoned>
// CHECK:STDOUT:     witness = (@ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.decl)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: %.loc19_6 as %AddWith.type {
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.decl: %i32.builtin.as.AddWith.impl.Op.type = fn_decl @i32.builtin.as.AddWith.impl.Op [concrete = constants.%i32.builtin.as.AddWith.impl.Op] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %other.param_patt: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:     %other.patt: %pattern_type.956 = at_binding_pattern other, %other.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %Self.ref.loc20_37 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Self.ref.loc20_37: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc20: Core.Form = init_form %Self.ref.loc20_37 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref.loc20_15: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = value_binding self, %self.param
// CHECK:STDOUT:     %other.param: %i32.builtin = value_param call_param1
// CHECK:STDOUT:     %Self.ref.loc20_28: type = name_ref Self, @i32.builtin.as.AddWith.impl.%.loc19_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %other: %i32.builtin = value_binding other, %other.param
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
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: %.loc23_17.2 as %As.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.decl: %Core.IntLiteral.as.As.impl.Convert.type = fn_decl @Core.IntLiteral.as.As.impl.Convert [concrete = constants.%Core.IntLiteral.as.As.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc24_31.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc24_31.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc24_31.2: Core.Form = init_form %.loc24_31.1 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.As.impl.%.loc23_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = value_binding self, %self.param
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
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: %.loc27_17.2 as %ImplicitAs.type {
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = fn_decl @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.dc0 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.dc0 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.956 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.956 = return_slot_pattern %return.param_patt, %.loc28_31.1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc28_31.1: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %.loc28_31.2: Core.Form = init_form %.loc28_31.1 [concrete = constants.%.9cb]
// CHECK:STDOUT:     %self.param: Core.IntLiteral = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @Core.IntLiteral.as.ImplicitAs.impl.%.loc27_17.2 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %self: Core.IntLiteral = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref %i32.builtin = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32.builtin = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%Core.IntLiteral.as.ImplicitAs.impl.Convert.decl), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.e40]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Convert = %Core.IntLiteral.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: %.loc31_6 as %ImplicitAs.type {
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.decl: %i32.builtin.as.ImplicitAs.impl.Convert.type = fn_decl @i32.builtin.as.ImplicitAs.impl.Convert [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.956 = value_param_pattern [concrete]
// CHECK:STDOUT:     %self.patt: %pattern_type.956 = at_binding_pattern self, %self.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.dc0 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.dc0 = return_slot_pattern %return.param_patt, %.loc32_42.2 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %IntLiteral.ref: %IntLiteral.type = name_ref IntLiteral, file.%IntLiteral.decl [concrete = constants.%IntLiteral]
// CHECK:STDOUT:     %IntLiteral.call: init type = call %IntLiteral.ref() [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.1: type = value_of_initializer %IntLiteral.call [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.2: type = converted %IntLiteral.call, %.loc32_42.1 [concrete = Core.IntLiteral]
// CHECK:STDOUT:     %.loc32_42.3: Core.Form = init_form %.loc32_42.2 [concrete = constants.%.f7d]
// CHECK:STDOUT:     %self.param: %i32.builtin = value_param call_param0
// CHECK:STDOUT:     %Self.ref: type = name_ref Self, @i32.builtin.as.ImplicitAs.impl.%.loc31_6 [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %self: %i32.builtin = value_binding self, %self.param
// CHECK:STDOUT:     %return.param: ref Core.IntLiteral = out_param call_param1
// CHECK:STDOUT:     %return: ref Core.IntLiteral = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table = impl_witness_table (%i32.builtin.as.ImplicitAs.impl.Convert.decl), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness: <witness> = impl_witness %ImplicitAs.impl_witness_table [concrete = constants.%ImplicitAs.impl_witness.57e]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .IntLiteral = <poisoned>
// CHECK:STDOUT:   .Convert = %i32.builtin.as.ImplicitAs.impl.Convert.decl
// CHECK:STDOUT:   extend %ImplicitAs.type
// CHECK:STDOUT:   witness = %ImplicitAs.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @IntLiteral() -> out %return.param: type = "int_literal.make_type";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int(%N.param: Core.IntLiteral) -> out %return.param: type = "int.make_type_signed";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(@AddWith.%T.loc7_20.2: type, @AddWith.%Self.loc7_29.1: @AddWith.%AddWith.type (%AddWith.type.adb)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.adb)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.1fa)]
// CHECK:STDOUT:   %Self.as_type.loc8_15.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc8_15.1 (constants.%Self.as_type.9bf)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type.loc8_15.1 [symbolic = %pattern_type (constants.%pattern_type.ad8)]
// CHECK:STDOUT:   %.loc8_37.1: Core.Form = init_form %Self.as_type.loc8_15.1 [symbolic = %.loc8_37.1 (constants.%.06b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf), %other.param: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf)) -> out %return.param: @AddWith.WithSelf.Op.%Self.as_type.loc8_15.1 (%Self.as_type.9bf);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(@As.%T.loc11_15.2: type, @As.%Self.loc11_24.1: @As.%As.type (%As.type.5df)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.5df)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.5df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.ddb)]
// CHECK:STDOUT:   %Self.as_type.loc12_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc12_20.1 (constants.%Self.as_type.784)]
// CHECK:STDOUT:   %pattern_type.loc12_18: type = pattern_type %Self.as_type.loc12_20.1 [symbolic = %pattern_type.loc12_18 (constants.%pattern_type.cb5)]
// CHECK:STDOUT:   %.loc12_31.1: Core.Form = init_form %T [symbolic = %.loc12_31.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc12_31: type = pattern_type %T [symbolic = %pattern_type.loc12_31 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @As.WithSelf.Convert.%Self.as_type.loc12_20.1 (%Self.as_type.784)) -> out %return.param: @As.WithSelf.Convert.%T (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(@ImplicitAs.%T.loc15_23.2: type, @ImplicitAs.%Self.loc15_32.1: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152)) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f53)]
// CHECK:STDOUT:   %Self.as_type.loc16_20.1: type = facet_access_type %Self [symbolic = %Self.as_type.loc16_20.1 (constants.%Self.as_type.423)]
// CHECK:STDOUT:   %pattern_type.loc16_18: type = pattern_type %Self.as_type.loc16_20.1 [symbolic = %pattern_type.loc16_18 (constants.%pattern_type.524)]
// CHECK:STDOUT:   %.loc16_31.1: Core.Form = init_form %T [symbolic = %.loc16_31.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc16_31: type = pattern_type %T [symbolic = %pattern_type.loc16_31 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param: @ImplicitAs.WithSelf.Convert.%Self.as_type.loc16_20.1 (%Self.as_type.423)) -> out %return.param: @ImplicitAs.WithSelf.Convert.%T (%T);
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
// CHECK:STDOUT:   %T.loc7_20.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.1fa) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.1fa) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.adb
// CHECK:STDOUT:   %Self => constants.%Self.1fa
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%Self.as_type.9bf
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ad8
// CHECK:STDOUT:   %.loc8_37.1 => constants.%.06b
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%T) {
// CHECK:STDOUT:   %T.loc11_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.ddb) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.ddb) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.5df
// CHECK:STDOUT:   %Self => constants.%Self.ddb
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => constants.%Self.as_type.784
// CHECK:STDOUT:   %pattern_type.loc12_18 => constants.%pattern_type.cb5
// CHECK:STDOUT:   %.loc12_31.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc12_31 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.f53) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.f53) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.152
// CHECK:STDOUT:   %Self => constants.%Self.f53
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%Self.as_type.423
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.524
// CHECK:STDOUT:   %.loc16_31.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc16_31 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc7_20.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self.loc7_29.2 => constants.%Self.52c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.1fa) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%Self.1fa
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.6e5
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.f8a
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.b08
// CHECK:STDOUT:   %assoc0.loc8_41.2 => constants.%assoc0.a97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.6cc
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.8ae
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.b08
// CHECK:STDOUT:   %assoc0.loc8_41.2 => constants.%assoc0.a97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %Self.as_type.loc8_15.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.956
// CHECK:STDOUT:   %.loc8_37.1 => constants.%.9cb
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc11_15.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self.loc11_24.2 => constants.%Self.2ec
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.ddb) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.ddb
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.ad6
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.e6e
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.0fc
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.1bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.dfe
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.d50
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.0fc
// CHECK:STDOUT:   %assoc0.loc12_32.2 => constants.%assoc0.1bf
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %Self.as_type.loc12_20.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc12_18 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %.loc12_31.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc12_31 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T.loc15_23.1 => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self.loc15_32.2 => constants.%Self.061
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.f53) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%Self.f53
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.7cb
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.ae1
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.aa1
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.715
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.b41) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.b41
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.2a8
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.b67
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.aa1
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.715
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%i32.builtin, constants.%ImplicitAs.facet.b41) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.b41
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => Core.IntLiteral
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.dc0
// CHECK:STDOUT:   %.loc16_31.1 => constants.%.9cb
// CHECK:STDOUT:   %pattern_type.loc16_31 => constants.%pattern_type.956
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T.loc15_23.1 => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self.loc15_32.2 => constants.%Self.98d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.f53) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%Self.f53
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.801
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.4dc
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.49c
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.549
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.529) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.529
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.3ed
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.d26
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.49c
// CHECK:STDOUT:   %assoc0.loc16_32.2 => constants.%assoc0.549
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(Core.IntLiteral, constants.%ImplicitAs.facet.529) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.529
// CHECK:STDOUT:   %Self.as_type.loc16_20.1 => constants.%i32.builtin
// CHECK:STDOUT:   %pattern_type.loc16_18 => constants.%pattern_type.956
// CHECK:STDOUT:   %.loc16_31.1 => constants.%.f7d
// CHECK:STDOUT:   %pattern_type.loc16_31 => constants.%pattern_type.dc0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- user.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [concrete]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %int_1.5b8: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %As.type.962: type = generic_interface_type @As [concrete]
// CHECK:STDOUT:   %As.generic: %As.type.962 = struct_value () [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %As.type.5df: type = facet_type <@As, @As(%T)> [symbolic]
// CHECK:STDOUT:   %Self.374: %As.type.5df = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.assoc_type.9d6: type = assoc_entity_type @As, @As(%T) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.4ab: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%T, %Self.374) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.786: %As.WithSelf.Convert.type.4ab = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Self.as_type.999: type = facet_access_type %Self.374 [symbolic]
// CHECK:STDOUT:   %pattern_type.ee4: type = pattern_type %Self.as_type.999 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %As.type.3aa: type = facet_type <@As, @As(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.05e: %As.type.3aa = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.f09: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %Self.374) [symbolic]
// CHECK:STDOUT:   %As.WithSelf.Convert.a8c: %As.WithSelf.Convert.type.f09 = struct_value () [symbolic]
// CHECK:STDOUT:   %As.assoc_type.0fc: type = assoc_entity_type @As, @As(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.812: %As.assoc_type.0fc = assoc_entity element0, imports.%Core.import_ref.e76 [concrete]
// CHECK:STDOUT:   %assoc0.6e4: %As.assoc_type.9d6 = assoc_entity element0, imports.%Core.import_ref.c91 [symbolic]
// CHECK:STDOUT:   %As.impl_witness: <witness> = impl_witness imports.%As.impl_witness_table [concrete]
// CHECK:STDOUT:   %As.facet: %As.type.3aa = facet_value Core.IntLiteral, (%As.impl_witness) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.type.058: type = fn_type @As.WithSelf.Convert, @As.WithSelf(%i32.builtin, %As.facet) [concrete]
// CHECK:STDOUT:   %As.WithSelf.Convert.9ec: %As.WithSelf.Convert.type.058 = struct_value () [concrete]
// CHECK:STDOUT:   %.a6a: type = fn_type_with_self_type %As.WithSelf.Convert.type.058, %As.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.type: type = fn_type @Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert: %Core.IntLiteral.as.As.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.2e9: <bound method> = bound_method %int_1.5b8, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_1.f38: %i32.builtin = int_value 1 [concrete]
// CHECK:STDOUT:   %int_2.ecc: Core.IntLiteral = int_value 2 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.9a5: <bound method> = bound_method %int_2.ecc, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_2.5a1: %i32.builtin = int_value 2 [concrete]
// CHECK:STDOUT:   %AddWith.type.5e0: type = generic_interface_type @AddWith [concrete]
// CHECK:STDOUT:   %AddWith.generic: %AddWith.type.5e0 = struct_value () [concrete]
// CHECK:STDOUT:   %AddWith.type.adb: type = facet_type <@AddWith, @AddWith(%T)> [symbolic]
// CHECK:STDOUT:   %Self.e74: %AddWith.type.adb = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.329: type = assoc_entity_type @AddWith, @AddWith(%T) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.171: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%T, %Self.e74) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.3fe: %AddWith.WithSelf.Op.type.171 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.f75: type = facet_access_type %Self.e74 [symbolic]
// CHECK:STDOUT:   %pattern_type.fb3: type = pattern_type %Self.as_type.f75 [symbolic]
// CHECK:STDOUT:   %.643: Core.Form = init_form %Self.as_type.f75 [symbolic]
// CHECK:STDOUT:   %AddWith.type.248: type = facet_type <@AddWith, @AddWith(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.191: %AddWith.type.248 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.a28: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %Self.e74) [symbolic]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.c76: %AddWith.WithSelf.Op.type.a28 = struct_value () [symbolic]
// CHECK:STDOUT:   %AddWith.assoc_type.b08: type = assoc_entity_type @AddWith, @AddWith(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.33a: %AddWith.assoc_type.b08 = assoc_entity element0, imports.%Core.import_ref.2b2 [concrete]
// CHECK:STDOUT:   %assoc0.cfb: %AddWith.assoc_type.329 = assoc_entity element0, imports.%Core.import_ref.8be [symbolic]
// CHECK:STDOUT:   %AddWith.impl_witness: <witness> = impl_witness imports.%AddWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %AddWith.facet: %AddWith.type.248 = facet_value %i32.builtin, (%AddWith.impl_witness) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type.98c: type = fn_type @AddWith.WithSelf.Op, @AddWith.WithSelf(%i32.builtin, %AddWith.facet) [concrete]
// CHECK:STDOUT:   %AddWith.WithSelf.Op.94a: %AddWith.WithSelf.Op.type.98c = struct_value () [concrete]
// CHECK:STDOUT:   %.ce3: type = fn_type_with_self_type %AddWith.WithSelf.Op.type.98c, %AddWith.facet [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.type: type = fn_type @i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op: %i32.builtin.as.AddWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.759: <bound method> = bound_method %int_1.f38, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_3.a0f: %i32.builtin = int_value 3 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.b23: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.b23 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.152: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic]
// CHECK:STDOUT:   %Self.f66: %ImplicitAs.type.152 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.6dd: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%T) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f65: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%T, %Self.f66) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.020: %ImplicitAs.WithSelf.Convert.type.f65 = struct_value () [symbolic]
// CHECK:STDOUT:   %Self.as_type.2c9: type = facet_access_type %Self.f66 [symbolic]
// CHECK:STDOUT:   %pattern_type.353: type = pattern_type %Self.as_type.2c9 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.ab8: type = facet_type <@ImplicitAs, @ImplicitAs(Core.IntLiteral)> [concrete]
// CHECK:STDOUT:   %Self.3bc: %ImplicitAs.type.ab8 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.02f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %Self.f66) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.bf5: %ImplicitAs.WithSelf.Convert.type.02f = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.49c: type = assoc_entity_type @ImplicitAs, @ImplicitAs(Core.IntLiteral) [concrete]
// CHECK:STDOUT:   %assoc0.b59: %ImplicitAs.assoc_type.49c = assoc_entity element0, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %assoc0.a16: %ImplicitAs.assoc_type.6dd = assoc_entity element0, imports.%Core.import_ref.b1d [symbolic]
// CHECK:STDOUT:   %ImplicitAs.type.c36: type = facet_type <@ImplicitAs, @ImplicitAs(%i32.builtin)> [concrete]
// CHECK:STDOUT:   %Self.1b0: %ImplicitAs.type.c36 = symbolic_binding Self, 1 [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.2a9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.70b [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.fec: %ImplicitAs.type.ab8 = facet_value %i32.builtin, (%ImplicitAs.impl_witness.2a9) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.f2b: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(Core.IntLiteral, %ImplicitAs.facet.fec) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.363: %ImplicitAs.WithSelf.Convert.type.f2b = struct_value () [concrete]
// CHECK:STDOUT:   %.2fd: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.f2b, %ImplicitAs.facet.fec [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.type: type = fn_type @i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert: %i32.builtin.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %int_3.1ba: Core.IntLiteral = int_value 3 [concrete]
// CHECK:STDOUT:   %array_type: type = array_type %int_3.1ba, %i32.builtin [concrete]
// CHECK:STDOUT:   %pattern_type.9e2: type = pattern_type %array_type [concrete]
// CHECK:STDOUT:   %int_4.0c1: Core.IntLiteral = int_value 4 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.1ac: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.bound.41b: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.As.impl.Convert [concrete]
// CHECK:STDOUT:   %int_4.4f1: %i32.builtin = int_value 4 [concrete]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.bound.c70: <bound method> = bound_method %int_3.a0f, %i32.builtin.as.AddWith.impl.Op [concrete]
// CHECK:STDOUT:   %int_7: %i32.builtin = int_value 7 [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (Core.IntLiteral, Core.IntLiteral, %i32.builtin) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%int_3.1ba, %int_4.0c1, %int_7) [concrete]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.978: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %Self.f66) [symbolic]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.fff: %ImplicitAs.WithSelf.Convert.type.978 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.assoc_type.aa1: type = assoc_entity_type @ImplicitAs, @ImplicitAs(%i32.builtin) [concrete]
// CHECK:STDOUT:   %assoc0.5ce: %ImplicitAs.assoc_type.aa1 = assoc_entity element0, imports.%Core.import_ref.b56 [concrete]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3b1: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.e02 [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet.dfd: %ImplicitAs.type.c36 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3b1) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.cca: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32.builtin, %ImplicitAs.facet.dfd) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.189: %ImplicitAs.WithSelf.Convert.type.cca = struct_value () [concrete]
// CHECK:STDOUT:   %.cee: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.cca, %ImplicitAs.facet.dfd [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1ac: <bound method> = bound_method %int_3.1ba, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.41b: <bound method> = bound_method %int_4.0c1, %Core.IntLiteral.as.ImplicitAs.impl.Convert [concrete]
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
// CHECK:STDOUT:   %Core.As: %As.type.962 = import_ref Core//default, As, loaded [concrete = constants.%As.generic]
// CHECK:STDOUT:   %Core.import_ref.eac: @As.WithSelf.%As.assoc_type (%As.assoc_type.9d6) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%assoc0 (constants.%assoc0.6e4)]
// CHECK:STDOUT:   %Core.Convert.90a = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.e76: @As.WithSelf.%As.WithSelf.Convert.type (%As.WithSelf.Convert.type.4ab) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.WithSelf.%As.WithSelf.Convert (constants.%As.WithSelf.Convert.786)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.4624d1.2: @As.%As.type (%As.type.5df) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%Self (constants.%Self.374)]
// CHECK:STDOUT:   %Core.import_ref.204 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @As.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.c91 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.945: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.497dad.1: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.bfc: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%As.type.3aa]
// CHECK:STDOUT:   %Core.import_ref.764ad9.1: %Core.IntLiteral.as.As.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %As.impl_witness_table = impl_witness_table (%Core.import_ref.764ad9.1), @Core.IntLiteral.as.As.impl [concrete]
// CHECK:STDOUT:   %Core.AddWith: %AddWith.type.5e0 = import_ref Core//default, AddWith, loaded [concrete = constants.%AddWith.generic]
// CHECK:STDOUT:   %Core.import_ref.a42: @AddWith.WithSelf.%AddWith.assoc_type (%AddWith.assoc_type.329) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%assoc0 (constants.%assoc0.cfb)]
// CHECK:STDOUT:   %Core.Op = import_ref Core//default, Op, unloaded
// CHECK:STDOUT:   %Core.import_ref.2b2: @AddWith.WithSelf.%AddWith.WithSelf.Op.type (%AddWith.WithSelf.Op.type.171) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.WithSelf.%AddWith.WithSelf.Op (constants.%AddWith.WithSelf.Op.3fe)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.5: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.b5f8a0.2: @AddWith.%AddWith.type (%AddWith.type.adb) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%Self (constants.%Self.e74)]
// CHECK:STDOUT:   %Core.import_ref.ec8 = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @AddWith.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.8be = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.0af: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.impl_witness]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.eb6: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%AddWith.type.248]
// CHECK:STDOUT:   %Core.import_ref.4c9: %i32.builtin.as.AddWith.impl.Op.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %AddWith.impl_witness_table = impl_witness_table (%Core.import_ref.4c9), @i32.builtin.as.AddWith.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.b23 = import_ref Core//default, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.f8a: @ImplicitAs.WithSelf.%ImplicitAs.assoc_type (%ImplicitAs.assoc_type.6dd) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%assoc0 (constants.%assoc0.a16)]
// CHECK:STDOUT:   %Core.Convert.466 = import_ref Core//default, Convert, unloaded
// CHECK:STDOUT:   %Core.import_ref.b56: @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert.type (%ImplicitAs.WithSelf.Convert.type.f65) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.WithSelf.%ImplicitAs.WithSelf.Convert (constants.%ImplicitAs.WithSelf.Convert.020)]
// CHECK:STDOUT:   %Core.import_ref.b3bc94.8: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.84c353.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%Self (constants.%Self.f66)]
// CHECK:STDOUT:   %Core.import_ref.79b = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b3bc94.9: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [symbolic = @ImplicitAs.%T (constants.%T)]
// CHECK:STDOUT:   %Core.import_ref.b1d = import_ref Core//default, loc{{\d+_\d+}}, unloaded
// CHECK:STDOUT:   %Core.import_ref.b33: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.3b1]
// CHECK:STDOUT:   %Core.import_ref.497dad.2: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = Core.IntLiteral]
// CHECK:STDOUT:   %Core.import_ref.197: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.c36]
// CHECK:STDOUT:   %Core.import_ref.863: <witness> = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.impl_witness.2a9]
// CHECK:STDOUT:   %Core.import_ref.63cbb1.3: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %Core.import_ref.87b: type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%ImplicitAs.type.ab8]
// CHECK:STDOUT:   %Core.import_ref.764ad9.2: %i32.builtin.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.70b = impl_witness_table (%Core.import_ref.764ad9.2), @i32.builtin.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.764ad9.3: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type = import_ref Core//default, loc{{\d+_\d+}}, loaded [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.e02 = impl_witness_table (%Core.import_ref.764ad9.3), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .arr = %arr
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %arr.patt: %pattern_type.9e2 = ref_binding_pattern arr [concrete]
// CHECK:STDOUT:     %arr.var_patt: %pattern_type.9e2 = var_pattern %arr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr.var: ref %array_type = var %arr.var_patt [concrete]
// CHECK:STDOUT:   %.loc4_44: type = splice_block %array_type [concrete = constants.%array_type] {
// CHECK:STDOUT:     %.loc4_16: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1.5b8]
// CHECK:STDOUT:     %.loc4_27: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_24: %.a6a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_24: <bound method> = bound_method %int_1, %impl.elem0.loc4_24 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.2e9]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_24: init %i32.builtin = call %bound_method.loc4_24(%int_1) [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_24 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %.loc4_24.2: %i32.builtin = converted %int_1, %.loc4_24.1 [concrete = constants.%int_1.f38]
// CHECK:STDOUT:     %int_2: Core.IntLiteral = int_value 2 [concrete = constants.%int_2.ecc]
// CHECK:STDOUT:     %.loc4_40: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:     %impl.elem0.loc4_37: %.a6a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_37: <bound method> = bound_method %int_2, %impl.elem0.loc4_37 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.9a5]
// CHECK:STDOUT:     %Core.IntLiteral.as.As.impl.Convert.call.loc4_37: init %i32.builtin = call %bound_method.loc4_37(%int_2) [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_37 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %.loc4_37.2: %i32.builtin = converted %int_2, %.loc4_37.1 [concrete = constants.%int_2.5a1]
// CHECK:STDOUT:     %impl.elem0.loc4_32.1: %.ce3 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:     %bound_method.loc4_32.1: <bound method> = bound_method %.loc4_24.2, %impl.elem0.loc4_32.1 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.759]
// CHECK:STDOUT:     %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_32.1(%.loc4_24.2, %.loc4_37.2) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %impl.elem0.loc4_32.2: %.2fd = impl_witness_access constants.%ImplicitAs.impl_witness.2a9, element0 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:     %bound_method.loc4_32.2: <bound method> = bound_method %i32.builtin.as.AddWith.impl.Op.call, %impl.elem0.loc4_32.2 [concrete = constants.%i32.builtin.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:     %.loc4_32.1: %i32.builtin = value_of_initializer %i32.builtin.as.AddWith.impl.Op.call [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %.loc4_32.2: %i32.builtin = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:     %i32.builtin.as.ImplicitAs.impl.Convert.call: init Core.IntLiteral = call %bound_method.loc4_32.2(%.loc4_32.2) [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.3: Core.IntLiteral = value_of_initializer %i32.builtin.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %.loc4_32.4: Core.IntLiteral = converted %i32.builtin.as.AddWith.impl.Op.call, %.loc4_32.3 [concrete = constants.%int_3.1ba]
// CHECK:STDOUT:     %array_type: type = array_type %.loc4_32.4, %.loc4_16 [concrete = constants.%array_type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %arr: ref %array_type = ref_binding arr, %arr.var [concrete = %arr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @As(imports.%Core.import_ref.b3bc94.3: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.5df)]
// CHECK:STDOUT:   %Self: @As.%As.type (%As.type.5df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.374)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.204
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.eac
// CHECK:STDOUT:     witness = (imports.%Core.Convert.90a)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @AddWith(imports.%Core.import_ref.b3bc94.6: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.adb)]
// CHECK:STDOUT:   %Self: @AddWith.%AddWith.type (%AddWith.type.adb) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e74)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.ec8
// CHECK:STDOUT:     .Op = imports.%Core.import_ref.a42
// CHECK:STDOUT:     witness = (imports.%Core.Op)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic interface @ImplicitAs(imports.%Core.import_ref.b3bc94.9: type) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f66)]
// CHECK:STDOUT:
// CHECK:STDOUT:   interface {
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = imports.%Core.import_ref.79b
// CHECK:STDOUT:     .Convert = imports.%Core.import_ref.f8a
// CHECK:STDOUT:     witness = (imports.%Core.Convert.466)
// CHECK:STDOUT:
// CHECK:STDOUT:   !requires:
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.As.impl: imports.%Core.import_ref.497dad.1 as imports.%Core.import_ref.bfc [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.945
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.AddWith.impl: imports.%Core.import_ref.63cbb1.2 as imports.%Core.import_ref.eb6 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.0af
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Core.IntLiteral.as.ImplicitAs.impl: imports.%Core.import_ref.497dad.2 as imports.%Core.import_ref.197 [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.b33
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @i32.builtin.as.ImplicitAs.impl: imports.%Core.import_ref.63cbb1.3 as imports.%Core.import_ref.87b [from "core.carbon"] {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = imports.%Core.import_ref.863
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Int = "int.make_type_signed" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @As.WithSelf.Convert(imports.%Core.import_ref.b3bc94.2: type, imports.%Core.import_ref.4624d1.2: @As.%As.type (%As.type.5df)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %As.type: type = facet_type <@As, @As(%T)> [symbolic = %As.type (constants.%As.type.5df)]
// CHECK:STDOUT:   %Self: @As.WithSelf.Convert.%As.type (%As.type.5df) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.374)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.999)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.ee4)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Core.IntLiteral.as.As.impl.Convert = "int.convert_checked" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @AddWith.WithSelf.Op(imports.%Core.import_ref.b3bc94.5: type, imports.%Core.import_ref.b5f8a0.2: @AddWith.%AddWith.type (%AddWith.type.adb)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %AddWith.type: type = facet_type <@AddWith, @AddWith(%T)> [symbolic = %AddWith.type (constants.%AddWith.type.adb)]
// CHECK:STDOUT:   %Self: @AddWith.WithSelf.Op.%AddWith.type (%AddWith.type.adb) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.e74)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.f75)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %Self.as_type [symbolic = %pattern_type (constants.%pattern_type.fb3)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %Self.as_type [symbolic = %.1 (constants.%.643)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @i32.builtin.as.AddWith.impl.Op = "int.sadd" [from "core.carbon"];
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @ImplicitAs.WithSelf.Convert(imports.%Core.import_ref.b3bc94.8: type, imports.%Core.import_ref.84c353.2: @ImplicitAs.%ImplicitAs.type (%ImplicitAs.type.152)) [from "core.carbon"] {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %ImplicitAs.type: type = facet_type <@ImplicitAs, @ImplicitAs(%T)> [symbolic = %ImplicitAs.type (constants.%ImplicitAs.type.152)]
// CHECK:STDOUT:   %Self: @ImplicitAs.WithSelf.Convert.%ImplicitAs.type (%ImplicitAs.type.152) = symbolic_binding Self, 1 [symbolic = %Self (constants.%Self.f66)]
// CHECK:STDOUT:   %Self.as_type: type = facet_access_type %Self [symbolic = %Self.as_type (constants.%Self.as_type.2c9)]
// CHECK:STDOUT:   %pattern_type.1: type = pattern_type %Self.as_type [symbolic = %pattern_type.1 (constants.%pattern_type.353)]
// CHECK:STDOUT:   %.1: Core.Form = init_form %T [symbolic = %.1 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.2: type = pattern_type %T [symbolic = %pattern_type.2 (constants.%pattern_type.51d)]
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
// CHECK:STDOUT:   %impl.elem0.loc4_58: %.a6a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_58: <bound method> = bound_method %int_3.loc4_56, %impl.elem0.loc4_58 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.1ac]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_58: init %i32.builtin = call %bound_method.loc4_58(%int_3.loc4_56) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_58 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_58.2: %i32.builtin = converted %int_3.loc4_56, %.loc4_58.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_4.loc4_69: Core.IntLiteral = int_value 4 [concrete = constants.%int_4.0c1]
// CHECK:STDOUT:   %.loc4_74: type = type_literal constants.%i32.builtin [concrete = constants.%i32.builtin]
// CHECK:STDOUT:   %impl.elem0.loc4_71: %.a6a = impl_witness_access constants.%As.impl_witness, element0 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_71: <bound method> = bound_method %int_4.loc4_69, %impl.elem0.loc4_71 [concrete = constants.%Core.IntLiteral.as.As.impl.Convert.bound.41b]
// CHECK:STDOUT:   %Core.IntLiteral.as.As.impl.Convert.call.loc4_71: init %i32.builtin = call %bound_method.loc4_71(%int_4.loc4_69) [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.1: %i32.builtin = value_of_initializer %Core.IntLiteral.as.As.impl.Convert.call.loc4_71 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %.loc4_71.2: %i32.builtin = converted %int_4.loc4_69, %.loc4_71.1 [concrete = constants.%int_4.4f1]
// CHECK:STDOUT:   %impl.elem0.loc4_66: %.ce3 = impl_witness_access constants.%AddWith.impl_witness, element0 [concrete = constants.%i32.builtin.as.AddWith.impl.Op]
// CHECK:STDOUT:   %bound_method.loc4_66: <bound method> = bound_method %.loc4_58.2, %impl.elem0.loc4_66 [concrete = constants.%i32.builtin.as.AddWith.impl.Op.bound.c70]
// CHECK:STDOUT:   %i32.builtin.as.AddWith.impl.Op.call: init %i32.builtin = call %bound_method.loc4_66(%.loc4_58.2, %.loc4_71.2) [concrete = constants.%int_7]
// CHECK:STDOUT:   %.loc4_78.1: %tuple.type = tuple_literal (%int_3.loc4_49, %int_4.loc4_52, %i32.builtin.as.AddWith.impl.Op.call) [concrete = constants.%tuple]
// CHECK:STDOUT:   %impl.elem0.loc4_78.1: %.cee = impl_witness_access constants.%ImplicitAs.impl_witness.3b1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.1: <bound method> = bound_method %int_3.loc4_49, %impl.elem0.loc4_78.1 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.1ac]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1: init %i32.builtin = call %bound_method.loc4_78.1(%int_3.loc4_49) [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %.loc4_78.2: init %i32.builtin = converted %int_3.loc4_49, %Core.IntLiteral.as.ImplicitAs.impl.Convert.call.loc4_78.1 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0]
// CHECK:STDOUT:   %.loc4_78.3: ref %i32.builtin = array_index file.%arr.var, %int_0
// CHECK:STDOUT:   %.loc4_78.4: init %i32.builtin to %.loc4_78.3 = in_place_init %.loc4_78.2 [concrete = constants.%int_3.a0f]
// CHECK:STDOUT:   %impl.elem0.loc4_78.2: %.cee = impl_witness_access constants.%ImplicitAs.impl_witness.3b1, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert]
// CHECK:STDOUT:   %bound_method.loc4_78.2: <bound method> = bound_method %int_4.loc4_52, %impl.elem0.loc4_78.2 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound.41b]
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
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%T, constants.%Self.374) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf.Convert(constants.%T, constants.%Self.374) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %As.type => constants.%As.type.5df
// CHECK:STDOUT:   %Self => constants.%Self.374
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.999
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.ee4
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.05e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%Self.374) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%Self.374
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.f09
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.a8c
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.0fc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.812
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @As.WithSelf(constants.%i32.builtin, constants.%As.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %As.type => constants.%As.type.3aa
// CHECK:STDOUT:   %Self => constants.%As.facet
// CHECK:STDOUT:   %As.WithSelf.Convert.type => constants.%As.WithSelf.Convert.type.058
// CHECK:STDOUT:   %As.WithSelf.Convert => constants.%As.WithSelf.Convert.9ec
// CHECK:STDOUT:   %As.assoc_type => constants.%As.assoc_type.0fc
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.812
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%T, constants.%Self.e74) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf.Op(constants.%T, constants.%Self.e74) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.adb
// CHECK:STDOUT:   %Self => constants.%Self.e74
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.f75
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fb3
// CHECK:STDOUT:   %.1 => constants.%.643
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%Self.191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%Self.e74) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%Self.e74
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.a28
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.c76
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.b08
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.33a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @AddWith.WithSelf(constants.%i32.builtin, constants.%AddWith.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %AddWith.type => constants.%AddWith.type.248
// CHECK:STDOUT:   %Self => constants.%AddWith.facet
// CHECK:STDOUT:   %AddWith.WithSelf.Op.type => constants.%AddWith.WithSelf.Op.type.98c
// CHECK:STDOUT:   %AddWith.WithSelf.Op => constants.%AddWith.WithSelf.Op.94a
// CHECK:STDOUT:   %AddWith.assoc_type => constants.%AddWith.assoc_type.b08
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.33a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%T, constants.%Self.f66) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf.Convert(constants.%T, constants.%Self.f66) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.152
// CHECK:STDOUT:   %Self => constants.%Self.f66
// CHECK:STDOUT:   %Self.as_type => constants.%Self.as_type.2c9
// CHECK:STDOUT:   %pattern_type.1 => constants.%pattern_type.353
// CHECK:STDOUT:   %.1 => constants.%.184
// CHECK:STDOUT:   %pattern_type.2 => constants.%pattern_type.51d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(Core.IntLiteral) {
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%Self.3bc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%Self.f66) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%Self.f66
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.02f
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.bf5
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.49c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs(constants.%i32.builtin) {
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%Self.1b0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(Core.IntLiteral, constants.%ImplicitAs.facet.fec) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => Core.IntLiteral
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.ab8
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.fec
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.f2b
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.363
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.49c
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.b59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%Self.f66) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%Self.f66
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.978
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.fff
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.aa1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @ImplicitAs.WithSelf(constants.%i32.builtin, constants.%ImplicitAs.facet.dfd) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T => constants.%i32.builtin
// CHECK:STDOUT:   %ImplicitAs.type => constants.%ImplicitAs.type.c36
// CHECK:STDOUT:   %Self => constants.%ImplicitAs.facet.dfd
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type => constants.%ImplicitAs.WithSelf.Convert.type.cca
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert => constants.%ImplicitAs.WithSelf.Convert.189
// CHECK:STDOUT:   %ImplicitAs.assoc_type => constants.%ImplicitAs.assoc_type.aa1
// CHECK:STDOUT:   %assoc0 => constants.%assoc0.5ce
// CHECK:STDOUT: }
// CHECK:STDOUT:
