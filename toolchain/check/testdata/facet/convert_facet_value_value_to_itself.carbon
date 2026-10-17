// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_itself.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_facet_value_value_to_itself.carbon

interface Animal {}

fn FeedAnimal[T: Animal](unused a: T) {}

fn HandleAnimal[T: Animal](a: T) { FeedAnimal(a); }

class Goat {}
impl Goat as Animal {}

fn F() {
  HandleAnimal({} as Goat);
}

// CHECK:STDOUT: --- convert_facet_value_value_to_itself.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.f00: %Animal.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.333: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.patt.e44: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.443: %Animal.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.01d: type = facet_access_type %T.443 [symbolic]
// CHECK:STDOUT:   %pattern_type.e90: type = pattern_type %T.as_type.01d [symbolic]
// CHECK:STDOUT:   %a.param_patt.84e: %pattern_type.e90 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %a.patt.febc60.1: %pattern_type.e90 = wrapper_binding_pattern a, %a.param_patt.84e [symbolic]
// CHECK:STDOUT:   %FeedAnimal.type: type = fn_type @FeedAnimal [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %FeedAnimal: %FeedAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.db4: <witness> = require_complete_type %T.as_type.01d [symbolic]
// CHECK:STDOUT:   %a.patt.febc60.2: %pattern_type.e90 = wrapper_binding_pattern a, %a.param_patt.84e [symbolic]
// CHECK:STDOUT:   %HandleAnimal.type: type = fn_type @HandleAnimal [concrete]
// CHECK:STDOUT:   %HandleAnimal: %HandleAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.1b3: <specific function> = specific_function %FeedAnimal, @FeedAnimal(%T.443) [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %.629: <witness> = impl_self_witness %Goat, @Animal [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness @Goat.as.Animal.impl.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.283: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %a.param_patt.8f5: %pattern_type.283 = value_param_pattern [concrete]
// CHECK:STDOUT:   %a.patt.fd6628.1: %pattern_type.283 = wrapper_binding_pattern a, %a.param_patt.8f5 [concrete]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal, @HandleAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT:   %.4ea: ref %Goat = temporary invalid, %Goat.val [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyImpl.type: type = facet_type <@DestroyImpl> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@DestroyImpl & @SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc25_19.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.934: %SubobjectDestroy.type = facet_value %Goat, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.c35: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.934) [concrete]
// CHECK:STDOUT:   %facet_value: %facet_type = facet_value %Goat, (%DestroyImpl.impl_witness.c35, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%facet_value) [concrete]
// CHECK:STDOUT:   %a.patt.fd6628.2: %pattern_type.283 = wrapper_binding_pattern a, %a.param_patt.8f5 [concrete]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.ea8: <specific function> = specific_function %FeedAnimal, @FeedAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     .SubobjectDestroy = %Core.SubobjectDestroy
// CHECK:STDOUT:     .DestroyImpl = %Core.DestroyImpl
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
// CHECK:STDOUT:   %Core.SubobjectDestroy: type = import_ref Core//prelude/parts/destroy, SubobjectDestroy, loaded [concrete = constants.%SubobjectDestroy.type]
// CHECK:STDOUT:   %Core.DestroyImpl: type = import_ref Core//prelude/parts/destroy, DestroyImpl, loaded [concrete = constants.%DestroyImpl.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .FeedAnimal = %FeedAnimal.decl
// CHECK:STDOUT:     .HandleAnimal = %HandleAnimal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %FeedAnimal.decl: %FeedAnimal.type = fn_decl @FeedAnimal [concrete = constants.%FeedAnimal] {
// CHECK:STDOUT:     %T.patt.loc17_16.1: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:     %a.param_patt.loc17_34.1: @FeedAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc17_34.2 (constants.%a.param_patt.84e)]
// CHECK:STDOUT:     %a.patt.loc17_34.1: @FeedAnimal.%pattern_type (%pattern_type.e90) = wrapper_binding_pattern a, %a.param_patt.loc17_34.1 [symbolic = %a.patt.loc17_34.2 (constants.%a.patt.febc60.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc17_18: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc17_16.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T.443)]
// CHECK:STDOUT:     %a.param: @FeedAnimal.%T.as_type.loc17_36.1 (%T.as_type.01d) = value_param call_param0
// CHECK:STDOUT:     %.loc17_36.1: type = splice_block %.loc17_36.2 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type.01d)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc17_16.2 [symbolic = %T.loc17_16.1 (constants.%T.443)]
// CHECK:STDOUT:       %T.as_type.loc17_36.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:       %.loc17_36.2: type = converted %T.ref, %T.as_type.loc17_36.2 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @FeedAnimal.%T.as_type.loc17_36.1 (%T.as_type.01d) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %HandleAnimal.decl: %HandleAnimal.type = fn_decl @HandleAnimal [concrete = constants.%HandleAnimal] {
// CHECK:STDOUT:     %T.patt.loc19_18.1: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_18.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:     %a.param_patt.loc19_29.1: @HandleAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc19_29.2 (constants.%a.param_patt.84e)]
// CHECK:STDOUT:     %a.patt.loc19_29.1: @HandleAnimal.%pattern_type (%pattern_type.e90) = wrapper_binding_pattern a, %a.param_patt.loc19_29.1 [symbolic = %a.patt.loc19_29.2 (constants.%a.patt.febc60.2)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc19_20: type = splice_block %Animal.ref [concrete = constants.%Animal.type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19_18.2: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc19_18.1 (constants.%T.443)]
// CHECK:STDOUT:     %a.param: @HandleAnimal.%T.as_type.loc19_31.1 (%T.as_type.01d) = value_param call_param0
// CHECK:STDOUT:     %.loc19_31.1: type = splice_block %.loc19_31.2 [symbolic = %T.as_type.loc19_31.1 (constants.%T.as_type.01d)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc19_18.2 [symbolic = %T.loc19_18.1 (constants.%T.443)]
// CHECK:STDOUT:       %T.as_type.loc19_31.2: type = facet_access_type %T.ref [symbolic = %T.as_type.loc19_31.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:       %.loc19_31.2: type = converted %T.ref, %T.as_type.loc19_31.2 [symbolic = %T.as_type.loc19_31.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @HandleAnimal.%T.as_type.loc19_31.1 (%T.as_type.01d) = wrapper_binding a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @Goat.as.Animal.impl [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc22: <witness> = impl_self_witness @Goat.as.Animal.impl.%Goat.ref, @Animal [concrete = constants.%.629]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = symbolic_binding Self, 0 [symbolic = constants.%Self.f00]
// CHECK:STDOUT:   %Animal.WithSelf.decl = interface_with_self_decl @Animal [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @Goat.as.Animal.impl: %Goat.ref as %Animal.ref {
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @Goat.as.Animal.impl [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Animal.ref
// CHECK:STDOUT:   witness = %Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @FeedAnimal(%T.loc17_16.2: %Animal.type) {
// CHECK:STDOUT:   %T.patt.loc17_16.2: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc17_16.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:   %T.loc17_16.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc17_16.1 (constants.%T.443)]
// CHECK:STDOUT:   %T.as_type.loc17_36.1: type = facet_access_type %T.loc17_16.1 [symbolic = %T.as_type.loc17_36.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc17_36.1 [symbolic = %pattern_type (constants.%pattern_type.e90)]
// CHECK:STDOUT:   %a.param_patt.loc17_34.2: @FeedAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc17_34.2 (constants.%a.param_patt.84e)]
// CHECK:STDOUT:   %a.patt.loc17_34.2: @FeedAnimal.%pattern_type (%pattern_type.e90) = wrapper_binding_pattern a, %a.param_patt.loc17_34.2 [symbolic = %a.patt.loc17_34.2 (constants.%a.patt.febc60.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc17_36.1 [symbolic = %require_complete (constants.%require_complete.db4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @FeedAnimal.%T.as_type.loc17_36.1 (%T.as_type.01d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @HandleAnimal(%T.loc19_18.2: %Animal.type) {
// CHECK:STDOUT:   %T.patt.loc19_18.2: %pattern_type.333 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc19_18.2 (constants.%T.patt.e44)]
// CHECK:STDOUT:   %T.loc19_18.1: %Animal.type = symbolic_binding T, 0 [symbolic = %T.loc19_18.1 (constants.%T.443)]
// CHECK:STDOUT:   %T.as_type.loc19_31.1: type = facet_access_type %T.loc19_18.1 [symbolic = %T.as_type.loc19_31.1 (constants.%T.as_type.01d)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc19_31.1 [symbolic = %pattern_type (constants.%pattern_type.e90)]
// CHECK:STDOUT:   %a.param_patt.loc19_29.2: @HandleAnimal.%pattern_type (%pattern_type.e90) = value_param_pattern [symbolic = %a.param_patt.loc19_29.2 (constants.%a.param_patt.84e)]
// CHECK:STDOUT:   %a.patt.loc19_29.2: @HandleAnimal.%pattern_type (%pattern_type.e90) = wrapper_binding_pattern a, %a.param_patt.loc19_29.2 [symbolic = %a.patt.loc19_29.2 (constants.%a.patt.febc60.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc19_31.1 [symbolic = %require_complete (constants.%require_complete.db4)]
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc19_36.2: <specific function> = specific_function constants.%FeedAnimal, @FeedAnimal(%T.loc19_18.1) [symbolic = %FeedAnimal.specific_fn.loc19_36.2 (constants.%FeedAnimal.specific_fn.1b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @HandleAnimal.%T.as_type.loc19_31.1 (%T.as_type.01d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %FeedAnimal.ref: %FeedAnimal.type = name_ref FeedAnimal, file.%FeedAnimal.decl [concrete = constants.%FeedAnimal]
// CHECK:STDOUT:     %a.ref: @HandleAnimal.%T.as_type.loc19_31.1 (%T.as_type.01d) = name_ref a, %a
// CHECK:STDOUT:     %.loc19_48.1: %Animal.type = converted constants.%T.as_type.01d, constants.%T.443 [symbolic = %T.loc19_18.1 (constants.%T.443)]
// CHECK:STDOUT:     %.loc19_48.2: %Animal.type = converted constants.%T.as_type.01d, constants.%T.443 [symbolic = %T.loc19_18.1 (constants.%T.443)]
// CHECK:STDOUT:     %FeedAnimal.specific_fn.loc19_36.1: <specific function> = specific_function %FeedAnimal.ref, @FeedAnimal(constants.%T.443) [symbolic = %FeedAnimal.specific_fn.loc19_36.2 (constants.%FeedAnimal.specific_fn.1b3)]
// CHECK:STDOUT:     %FeedAnimal.call: init %empty_tuple.type = call %FeedAnimal.specific_fn.loc19_36.1(%a.ref)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %HandleAnimal.ref: %HandleAnimal.type = name_ref HandleAnimal, file.%HandleAnimal.decl [concrete = constants.%HandleAnimal]
// CHECK:STDOUT:   %.loc25_17.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc25_17.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc25_17.3: init %Goat to %.loc25_17.2 = class_init () [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc25_19.1: init %Goat = converted %.loc25_17.1, %.loc25_17.3 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %Animal.facet.loc25_26.1: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc25_26.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc25_26.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc25_26.2: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc25_26.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc25_26.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %HandleAnimal.specific_fn: <specific function> = specific_function %HandleAnimal.ref, @HandleAnimal(constants.%Animal.facet) [concrete = constants.%HandleAnimal.specific_fn]
// CHECK:STDOUT:   %.loc25_19.2: ref %Goat = temporary %.loc25_17.2, %.loc25_19.1 [concrete = constants.%.4ea]
// CHECK:STDOUT:   %.loc25_19.3: %Goat = acquire_value %.loc25_19.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %HandleAnimal.call: init %empty_tuple.type = call %HandleAnimal.specific_fn(%.loc25_19.3)
// CHECK:STDOUT:   %facet_value.loc25_19.1: %facet_type = facet_value constants.%Goat, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc25_19.4: %facet_type = converted constants.%Goat, %facet_value.loc25_19.1 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %facet_value.loc25_19.2: %facet_type = facet_value constants.%Goat, (constants.%DestroyImpl.impl_witness.c35, constants.%custom_witness.b80ef6.2) [concrete = constants.%facet_value]
// CHECK:STDOUT:   %.loc25_19.5: %facet_type = converted constants.%Goat, %facet_value.loc25_19.2 [concrete = constants.%facet_value]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%facet_value) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc25_19.2)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_19.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc25_19.2(%self.param: ref %Goat) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Self.f00) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%T.443) {
// CHECK:STDOUT:   %T.patt.loc17_16.2 => constants.%T.patt.e44
// CHECK:STDOUT:   %T.loc17_16.1 => constants.%T.443
// CHECK:STDOUT:   %T.as_type.loc17_36.1 => constants.%T.as_type.01d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e90
// CHECK:STDOUT:   %a.param_patt.loc17_34.2 => constants.%a.param_patt.84e
// CHECK:STDOUT:   %a.patt.loc17_34.2 => constants.%a.patt.febc60.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.db4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%T.443) {
// CHECK:STDOUT:   %T.patt.loc19_18.2 => constants.%T.patt.e44
// CHECK:STDOUT:   %T.loc19_18.1 => constants.%T.443
// CHECK:STDOUT:   %T.as_type.loc19_31.1 => constants.%T.as_type.01d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.e90
// CHECK:STDOUT:   %a.param_patt.loc19_29.2 => constants.%a.param_patt.84e
// CHECK:STDOUT:   %a.patt.loc19_29.2 => constants.%a.patt.febc60.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Animal.WithSelf(constants.%Animal.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @HandleAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.patt.loc19_18.2 => constants.%T.patt.e44
// CHECK:STDOUT:   %T.loc19_18.1 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc19_31.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.283
// CHECK:STDOUT:   %a.param_patt.loc19_29.2 => constants.%a.param_patt.8f5
// CHECK:STDOUT:   %a.patt.loc19_29.2 => constants.%a.patt.fd6628.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %FeedAnimal.specific_fn.loc19_36.2 => constants.%FeedAnimal.specific_fn.ea8
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @FeedAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.patt.loc17_16.2 => constants.%T.patt.e44
// CHECK:STDOUT:   %T.loc17_16.1 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc17_36.1 => constants.%Goat
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.283
// CHECK:STDOUT:   %a.param_patt.loc17_34.2 => constants.%a.param_patt.8f5
// CHECK:STDOUT:   %a.patt.loc17_34.2 => constants.%a.patt.fd6628.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
