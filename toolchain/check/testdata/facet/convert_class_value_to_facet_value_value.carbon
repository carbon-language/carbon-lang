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
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/facet/convert_class_value_to_facet_value_value.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/facet/convert_class_value_to_facet_value_value.carbon

interface Animal {}

fn WalkAnimal[T:! Animal](a: T) {}

class Goat {}
impl Goat as Animal {}

fn F() {
  WalkAnimal({} as Goat);
}

// CHECK:STDOUT: --- convert_class_value_to_facet_value_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Animal.type: type = facet_type <@Animal> [concrete]
// CHECK:STDOUT:   %Self.fd4: %Animal.type = bind_symbolic_name Self, 0 [symbolic]
// CHECK:STDOUT:   %T.fd4: %Animal.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b0: type = pattern_type %Animal.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.fd4 [symbolic]
// CHECK:STDOUT:   %pattern_type.36a: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %WalkAnimal.type: type = fn_type @WalkAnimal [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %WalkAnimal: %WalkAnimal.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.234: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Goat: type = class_type @Goat [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness file.%Animal.impl_witness_table [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %Goat.val: %Goat = struct_value () [concrete]
// CHECK:STDOUT:   %Animal.facet: %Animal.type = facet_value %Goat, (%Animal.impl_witness) [concrete]
// CHECK:STDOUT:   %pattern_type.ab7: type = pattern_type %Goat [concrete]
// CHECK:STDOUT:   %WalkAnimal.specific_fn: <specific function> = specific_function %WalkAnimal, @WalkAnimal(%Animal.facet) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Op.type.bae: type = fn_type @Op.1 [concrete]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Op.type.bc9: type = fn_type @Op.2, @impl.49c(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Op.46f: %Op.type.bc9 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.b14: <witness> = impl_witness imports.%Destroy.impl_witness_table, @impl.49c(%Goat) [concrete]
// CHECK:STDOUT:   %Op.type.e7a: type = fn_type @Op.2, @impl.49c(%Goat) [concrete]
// CHECK:STDOUT:   %Op.55a: %Op.type.e7a = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.940: type = ptr_type %Goat [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Goat, (%Destroy.impl_witness.b14) [concrete]
// CHECK:STDOUT:   %.095: type = fn_type_with_self_type %Op.type.bae, %Destroy.facet [concrete]
// CHECK:STDOUT:   %Op.specific_fn: <specific function> = specific_function %Op.55a, @Op.2(%Goat) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.0b9: @impl.49c.%Op.type (%Op.type.bc9) = import_ref Core//prelude/parts/destroy, loc8_29, loaded [symbolic = @impl.49c.%Op (constants.%Op.46f)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.0b9), @impl.49c [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Animal = %Animal.decl
// CHECK:STDOUT:     .WalkAnimal = %WalkAnimal.decl
// CHECK:STDOUT:     .Goat = %Goat.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Animal.decl: type = interface_decl @Animal [concrete = constants.%Animal.type] {} {}
// CHECK:STDOUT:   %WalkAnimal.decl: %WalkAnimal.type = fn_decl @WalkAnimal [concrete = constants.%WalkAnimal] {
// CHECK:STDOUT:     %T.patt: %pattern_type.3b0 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %a.patt: @WalkAnimal.%pattern_type (%pattern_type.36a) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @WalkAnimal.%pattern_type (%pattern_type.36a) = value_param_pattern %a.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:     %T.loc17_15.1: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.2 (constants.%T.fd4)]
// CHECK:STDOUT:     %a.param: @WalkAnimal.%T.as_type.loc17_30.2 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc17_30.1: type = splice_block %.loc17_30.2 [symbolic = %T.as_type.loc17_30.2 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref: %Animal.type = name_ref T, %T.loc17_15.1 [symbolic = %T.loc17_15.2 (constants.%T.fd4)]
// CHECK:STDOUT:       %T.as_type.loc17_30.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc17_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc17_30.2: type = converted %T.ref, %T.as_type.loc17_30.1 [symbolic = %T.as_type.loc17_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %a: @WalkAnimal.%T.as_type.loc17_30.2 (%T.as_type) = bind_name a, %a.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Goat.decl: type = class_decl @Goat [concrete = constants.%Goat] {} {}
// CHECK:STDOUT:   impl_decl @impl.27e [concrete] {} {
// CHECK:STDOUT:     %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:     %Animal.ref: type = name_ref Animal, file.%Animal.decl [concrete = constants.%Animal.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Animal.impl_witness_table = impl_witness_table (), @impl.27e [concrete]
// CHECK:STDOUT:   %Animal.impl_witness: <witness> = impl_witness %Animal.impl_witness_table [concrete = constants.%Animal.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Animal {
// CHECK:STDOUT:   %Self: %Animal.type = bind_symbolic_name Self, 0 [symbolic = constants.%Self.fd4]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @impl.27e: %Goat.ref as %Animal.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Animal.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @Goat {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%Goat
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WalkAnimal(%T.loc17_15.1: %Animal.type) {
// CHECK:STDOUT:   %T.loc17_15.2: %Animal.type = bind_symbolic_name T, 0 [symbolic = %T.loc17_15.2 (constants.%T.fd4)]
// CHECK:STDOUT:   %T.as_type.loc17_30.2: type = facet_access_type %T.loc17_15.2 [symbolic = %T.as_type.loc17_30.2 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc17_30.2 [symbolic = %pattern_type (constants.%pattern_type.36a)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc17_30.2 [symbolic = %require_complete (constants.%require_complete.234)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%a.param: @WalkAnimal.%T.as_type.loc17_30.2 (%T.as_type)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %WalkAnimal.ref: %WalkAnimal.type = name_ref WalkAnimal, file.%WalkAnimal.decl [concrete = constants.%WalkAnimal]
// CHECK:STDOUT:   %.loc23_15.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %Goat.ref: type = name_ref Goat, file.%Goat.decl [concrete = constants.%Goat]
// CHECK:STDOUT:   %.loc23_15.2: ref %Goat = temporary_storage
// CHECK:STDOUT:   %.loc23_15.3: init %Goat = class_init (), %.loc23_15.2 [concrete = constants.%Goat.val]
// CHECK:STDOUT:   %.loc23_15.4: ref %Goat = temporary %.loc23_15.2, %.loc23_15.3
// CHECK:STDOUT:   %.loc23_17.1: ref %Goat = converted %.loc23_15.1, %.loc23_15.4
// CHECK:STDOUT:   %Animal.facet.loc23_24.1: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc23_24.1: %Animal.type = converted constants.%Goat, %Animal.facet.loc23_24.1 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %Animal.facet.loc23_24.2: %Animal.type = facet_value constants.%Goat, (constants.%Animal.impl_witness) [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %.loc23_24.2: %Animal.type = converted constants.%Goat, %Animal.facet.loc23_24.2 [concrete = constants.%Animal.facet]
// CHECK:STDOUT:   %WalkAnimal.specific_fn: <specific function> = specific_function %WalkAnimal.ref, @WalkAnimal(constants.%Animal.facet) [concrete = constants.%WalkAnimal.specific_fn]
// CHECK:STDOUT:   %.loc23_17.2: %Goat = bind_value %.loc23_17.1
// CHECK:STDOUT:   %WalkAnimal.call: init %empty_tuple.type = call %WalkAnimal.specific_fn(%.loc23_17.2)
// CHECK:STDOUT:   %impl.elem0: %.095 = impl_witness_access constants.%Destroy.impl_witness.b14, element0 [concrete = constants.%Op.55a]
// CHECK:STDOUT:   %bound_method.loc23_15.1: <bound method> = bound_method %.loc23_15.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Op.2(constants.%Goat) [concrete = constants.%Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc23_15.2: <bound method> = bound_method %.loc23_15.2, %specific_fn
// CHECK:STDOUT:   %addr: %ptr.940 = addr_of %.loc23_15.2
// CHECK:STDOUT:   %no_op: init %empty_tuple.type = call %bound_method.loc23_15.2(%addr)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkAnimal(constants.%T.fd4) {
// CHECK:STDOUT:   %T.loc17_15.2 => constants.%T.fd4
// CHECK:STDOUT:   %T.as_type.loc17_30.2 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.36a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WalkAnimal(constants.%Animal.facet) {
// CHECK:STDOUT:   %T.loc17_15.2 => constants.%Animal.facet
// CHECK:STDOUT:   %T.as_type.loc17_30.2 => constants.%Goat
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ab7
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
