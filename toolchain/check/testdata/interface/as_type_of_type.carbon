// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interface/as_type_of_type.carbon

interface Empty {}

class C {}

impl C as Empty {}

fn F(T:! Empty) {
  var unused x: T*;
}

// CHECK:STDOUT: --- as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.61e: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness @C.as.Empty.impl.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %Empty.facet: %Empty.type = facet_value %C, (%Empty.impl_witness) [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.7d1: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %T.0c3: %Empty.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.as_type.d6c: type = facet_access_type %T.0c3 [symbolic]
// CHECK:STDOUT:   %ptr: type = ptr_type %T.as_type.d6c [symbolic]
// CHECK:STDOUT:   %require_complete.279: <witness> = require_complete_type %ptr [symbolic]
// CHECK:STDOUT:   %pattern_type.91f: type = pattern_type %ptr [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @DefaultOrUnformed [symbolic]
// CHECK:STDOUT:   %.4ce: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%ptr) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.facet: %DefaultOrUnformed.type = facet_value %ptr, (%DefaultOrUnformed.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.1a2: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet) [symbolic]
// CHECK:STDOUT:   %.4c7: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.1a2, %DefaultOrUnformed.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.00f: %.4c7 = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2bc: <specific function> = specific_impl_function %impl.elem0.00f, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d94: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic]
// CHECK:STDOUT:   %.987: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d94, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %impl.elem0.608: %.987 = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.2ba: <specific function> = specific_impl_function %impl.elem0.608, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .DefaultOrUnformed = %Core.DefaultOrUnformed
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.DefaultOrUnformed: type = import_ref Core//prelude/parts/default, DefaultOrUnformed, loaded [concrete = constants.%DefaultOrUnformed.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Empty = %Empty.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Empty.decl: type = interface_decl @Empty [concrete = constants.%Empty.type] {} {}
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   impl_decl @C.as.Empty.impl [concrete] {} {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.7d1 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_7.2: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.0c3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.61e]
// CHECK:STDOUT:   %Empty.WithSelf.decl = interface_with_self_decl @Empty [concrete]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Empty.impl: %C.ref as %Empty.ref {
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @C.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   extend %Empty.ref
// CHECK:STDOUT:   witness = %Empty.impl_witness
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc21_7.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc21_7.1: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc21_7.1 (constants.%T.0c3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.as_type.loc22_18.2: type = facet_access_type %T.loc21_7.1 [symbolic = %T.as_type.loc22_18.2 (constants.%T.as_type.d6c)]
// CHECK:STDOUT:   %ptr.loc22_18.2: type = ptr_type %T.as_type.loc22_18.2 [symbolic = %ptr.loc22_18.2 (constants.%ptr)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %ptr.loc22_18.2 [symbolic = %require_complete (constants.%require_complete.279)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc22_18.2 [symbolic = %pattern_type (constants.%pattern_type.91f)]
// CHECK:STDOUT:   %.loc22_19.3: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%ptr.loc22_18.2) [symbolic = %.loc22_19.3 (constants.%.4ce)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc22_18.2, @DefaultOrUnformed [symbolic = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc22_19.2: %DefaultOrUnformed.type = facet_value %ptr.loc22_18.2, (%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc22_19.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.loc22_19.2) [symbolic = %DefaultOrUnformed.WithSelf.Op.type (constants.%DefaultOrUnformed.WithSelf.Op.type.1a2)]
// CHECK:STDOUT:   %.loc22_19.4: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type, %DefaultOrUnformed.facet.loc22_19.2 [symbolic = %.loc22_19.4 (constants.%.4c7)]
// CHECK:STDOUT:   %impl.elem0.loc22_19.2: @F.%.loc22_19.4 (%.4c7) = impl_witness_access %DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_19.2 (constants.%impl.elem0.00f)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_19.2: <specific function> = specific_impl_function %impl.elem0.loc22_19.2, @DefaultOrUnformed.WithSelf.Op(%DefaultOrUnformed.facet.loc22_19.2) [symbolic = %specific_impl_fn.loc22_19.2 (constants.%specific_impl_fn.2bc)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc22_18.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc22_18.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d94)]
// CHECK:STDOUT:   %.loc22_3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc22_3 (constants.%.987)]
// CHECK:STDOUT:   %impl.elem0.loc22_3.2: @F.%.loc22_3 (%.987) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.608)]
// CHECK:STDOUT:   %specific_impl_fn.loc22_3.2: <specific function> = specific_impl_function %impl.elem0.loc22_3.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.2ba)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: @F.%pattern_type (%pattern_type.91f) = ref_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: @F.%pattern_type (%pattern_type.91f) = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref @F.%ptr.loc22_18.2 (%ptr) = var %x.var_patt
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc22_19.1: %DefaultOrUnformed.type = facet_value constants.%ptr, (constants.%DefaultOrUnformed.lookup_impl_witness) [symbolic = %DefaultOrUnformed.facet.loc22_19.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:     %.loc22_19.1: %DefaultOrUnformed.type = converted constants.%ptr, %DefaultOrUnformed.facet.loc22_19.1 [symbolic = %DefaultOrUnformed.facet.loc22_19.2 (constants.%DefaultOrUnformed.facet)]
// CHECK:STDOUT:     %as_type: type = facet_access_type %.loc22_19.1 [symbolic = %ptr.loc22_18.2 (constants.%ptr)]
// CHECK:STDOUT:     %.loc22_19.2: type = converted %.loc22_19.1, %as_type [symbolic = %ptr.loc22_18.2 (constants.%ptr)]
// CHECK:STDOUT:     %impl.elem0.loc22_19.1: @F.%.loc22_19.4 (%.4c7) = impl_witness_access constants.%DefaultOrUnformed.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_19.2 (constants.%impl.elem0.00f)]
// CHECK:STDOUT:     %specific_impl_fn.loc22_19.1: <specific function> = specific_impl_function %impl.elem0.loc22_19.1, @DefaultOrUnformed.WithSelf.Op(constants.%DefaultOrUnformed.facet) [symbolic = %specific_impl_fn.loc22_19.2 (constants.%specific_impl_fn.2bc)]
// CHECK:STDOUT:     %DefaultOrUnformed.WithSelf.Op.call: init @F.%ptr.loc22_18.2 (%ptr) = call %specific_impl_fn.loc22_19.1()
// CHECK:STDOUT:     assign %x.var, %DefaultOrUnformed.WithSelf.Op.call
// CHECK:STDOUT:     %.loc22_18.1: type = splice_block %ptr.loc22_18.1 [symbolic = %ptr.loc22_18.2 (constants.%ptr)] {
// CHECK:STDOUT:       %T.ref: %Empty.type = name_ref T, %T.loc21_7.2 [symbolic = %T.loc21_7.1 (constants.%T.0c3)]
// CHECK:STDOUT:       %T.as_type.loc22_18.1: type = facet_access_type %T.ref [symbolic = %T.as_type.loc22_18.2 (constants.%T.as_type.d6c)]
// CHECK:STDOUT:       %.loc22_18.2: type = converted %T.ref, %T.as_type.loc22_18.1 [symbolic = %T.as_type.loc22_18.2 (constants.%T.as_type.d6c)]
// CHECK:STDOUT:       %ptr.loc22_18.1: type = ptr_type %.loc22_18.2 [symbolic = %ptr.loc22_18.2 (constants.%ptr)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @F.%ptr.loc22_18.2 (%ptr) = ref_binding x, %x.var
// CHECK:STDOUT:     %impl.elem0.loc22_3.1: @F.%.loc22_3 (%.987) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22_3.2 (constants.%impl.elem0.608)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %x.var, %impl.elem0.loc22_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc22_3.1: <specific function> = specific_impl_function %impl.elem0.loc22_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet) [symbolic = %specific_impl_fn.loc22_3.2 (constants.%specific_impl_fn.2ba)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %x.var, %specific_impl_fn.loc22_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc22_3.2(%x.var)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Self.61e) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Empty.WithSelf(constants.%Empty.facet) {
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.0c3) {
// CHECK:STDOUT:   %T.loc21_7.1 => constants.%T.0c3
// CHECK:STDOUT: }
// CHECK:STDOUT:
