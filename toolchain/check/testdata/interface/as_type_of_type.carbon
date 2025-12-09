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
  var x: T*;
}

// CHECK:STDOUT: --- as_type_of_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Empty.type: type = facet_type <@Empty> [concrete]
// CHECK:STDOUT:   %Self.2a7: %Empty.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness file.%Empty.impl_witness_table [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: %Empty.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.639: type = pattern_type %Empty.type [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T [symbolic]
// CHECK:STDOUT:   %ptr.ac9: type = ptr_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %require_complete.78f: <witness> = require_complete_type %ptr.ac9 [symbolic]
// CHECK:STDOUT:   %pattern_type.0e5: type = pattern_type %ptr.ac9 [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %DestroyT: %type_where = symbolic_binding DestroyT, 0 [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%DestroyT) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fd5: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c = struct_value () [symbolic]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.ac9, () [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.080: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.13d: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.f7c: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.13d = struct_value () [symbolic]
// CHECK:STDOUT:   %ptr.d9e: type = ptr_type %ptr.ac9 [symbolic]
// CHECK:STDOUT:   %require_complete.ff7: <witness> = require_complete_type %ptr.d9e [symbolic]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.ac9, (%Destroy.impl_witness.080) [symbolic]
// CHECK:STDOUT:   %.c8c: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet [symbolic]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.f7c, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.import_ref.5c4: @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.15c) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @DestroyT.binding.as_type.as.Destroy.impl.%DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fd5)]
// CHECK:STDOUT:   %Destroy.impl_witness_table = impl_witness_table (%Core.import_ref.5c4), @DestroyT.binding.as_type.as.Destroy.impl [concrete]
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
// CHECK:STDOUT:   %Empty.impl_witness_table = impl_witness_table (), @C.as.Empty.impl [concrete]
// CHECK:STDOUT:   %Empty.impl_witness: <witness> = impl_witness %Empty.impl_witness_table [concrete = constants.%Empty.impl_witness]
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.639 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc21: type = splice_block %Empty.ref [concrete = constants.%Empty.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Empty.ref: type = name_ref Empty, file.%Empty.decl [concrete = constants.%Empty.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21_6.2: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: interface @Empty {
// CHECK:STDOUT:   %Self: %Empty.type = symbolic_binding Self, 0 [symbolic = constants.%Self.2a7]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = %Self
// CHECK:STDOUT:   witness = ()
// CHECK:STDOUT:
// CHECK:STDOUT: !requires:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: impl @C.as.Empty.impl: %C.ref as %Empty.ref {
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   witness = file.%Empty.impl_witness
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
// CHECK:STDOUT: generic fn @F(%T.loc21_6.2: %Empty.type) {
// CHECK:STDOUT:   %T.loc21_6.1: %Empty.type = symbolic_binding T, 0 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc21_6.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %ptr.loc22_11.2: type = ptr_type %T.binding.as_type [symbolic = %ptr.loc22_11.2 (constants.%ptr.ac9)]
// CHECK:STDOUT:   %require_complete.loc22_11: <witness> = require_complete_type %ptr.loc22_11.2 [symbolic = %require_complete.loc22_11 (constants.%require_complete.78f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %ptr.loc22_11.2 [symbolic = %pattern_type (constants.%pattern_type.0e5)]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %ptr.loc22_11.2, () [symbolic = %facet_value (constants.%facet_value)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness imports.%Destroy.impl_witness_table, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.080)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %ptr.loc22_11.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet)]
// CHECK:STDOUT:   %.loc22_3: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc22_3 (constants.%.c8c)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.type (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.type.13d)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op: @F.%DestroyT.binding.as_type.as.Destroy.impl.Op.type (%DestroyT.binding.as_type.as.Destroy.impl.Op.type.13d) = struct_value () [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f7c)]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:   %ptr.loc22_3: type = ptr_type %ptr.loc22_11.2 [symbolic = %ptr.loc22_3 (constants.%ptr.d9e)]
// CHECK:STDOUT:   %require_complete.loc22_3: <witness> = require_complete_type %ptr.loc22_3 [symbolic = %require_complete.loc22_3 (constants.%require_complete.ff7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %x.patt: @F.%pattern_type (%pattern_type.0e5) = ref_binding_pattern x [concrete]
// CHECK:STDOUT:       %x.var_patt: @F.%pattern_type (%pattern_type.0e5) = var_pattern %x.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x.var: ref @F.%ptr.loc22_11.2 (%ptr.ac9) = var %x.var_patt
// CHECK:STDOUT:     %.loc22_11.1: type = splice_block %ptr.loc22_11.1 [symbolic = %ptr.loc22_11.2 (constants.%ptr.ac9)] {
// CHECK:STDOUT:       %T.ref: %Empty.type = name_ref T, %T.loc21_6.2 [symbolic = %T.loc21_6.1 (constants.%T)]
// CHECK:STDOUT:       %T.as_type: type = facet_access_type %T.ref [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc22_11.2: type = converted %T.ref, %T.as_type [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %ptr.loc22_11.1: type = ptr_type %.loc22_11.2 [symbolic = %ptr.loc22_11.2 (constants.%ptr.ac9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: ref @F.%ptr.loc22_11.2 (%ptr.ac9) = ref_binding x, %x.var
// CHECK:STDOUT:     %impl.elem0: @F.%.loc22_3 (%.c8c) = impl_witness_access constants.%Destroy.impl_witness.080, element0 [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.f7c)]
// CHECK:STDOUT:     %bound_method.loc22_3.1: <bound method> = bound_method %x.var, %impl.elem0
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [symbolic = %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn (constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn)]
// CHECK:STDOUT:     %bound_method.loc22_3.2: <bound method> = bound_method %x.var, %specific_fn
// CHECK:STDOUT:     %addr: @F.%ptr.loc22_3 (%ptr.d9e) = addr_of %x.var
// CHECK:STDOUT:     %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc22_3.2(%addr)
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc21_6.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
