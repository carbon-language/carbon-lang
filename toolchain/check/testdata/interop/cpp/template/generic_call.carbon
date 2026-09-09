// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// EXTRA-ARGS: --clang-arg=--std=c++20
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/template/generic_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/template/generic_call.carbon

// --- header.h

struct X {};

template<typename T> struct S {};

// --- generic_call.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

//@dump-sem-ir-begin
fn F[T: type](unused x: T) {
  var unused v: Cpp.S(T);
}
//@dump-sem-ir-end

fn G() {
  F({} as Cpp.X);
}

// CHECK:STDOUT: --- generic_call.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.d47: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.51d1c4.1: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d1c4.1 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d1c4.1 = wrapper_binding_pattern x, %x.param_patt.91d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %S.type: type = cpp_type_template_type S [concrete]
// CHECK:STDOUT:   %S.template: %S.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a92: type = template_inst %T.67d [template]
// CHECK:STDOUT:   %.4a6: type = splice_inst @F.%.loc8_24.6 [template]
// CHECK:STDOUT:   %.52a: type = type_of_inst @F.%.loc8_24.8 [template]
// CHECK:STDOUT:   %.2a3: %.52a = splice_inst @F.%.loc8_24.8 [template]
// CHECK:STDOUT:   %.8bd: type = splice_inst @F.%.loc8_24.11 [template]
// CHECK:STDOUT:   %require_complete.cfd: <witness> = require_complete_type %.8bd [template]
// CHECK:STDOUT:   %pattern_type.0fb: type = pattern_type %.8bd [template]
// CHECK:STDOUT:   %v.patt.5bf: %pattern_type.0fb = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.2d3: %pattern_type.0fb = var_pattern %v.patt.5bf [template]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Self.935: %DefaultOrUnformed.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.8f1: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%Self.935) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.ad7: %DefaultOrUnformed.WithSelf.Op.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.8bd, @DefaultOrUnformed [template]
// CHECK:STDOUT:   %.892: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.8bd) [template]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.77c: %DefaultOrUnformed.type = facet_value %.8bd, (%DefaultOrUnformed.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.92d: %DefaultOrUnformed.type = splice_inst @F.%.loc8_25.8 [template]
// CHECK:STDOUT:   %.177: type = type_of_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %.f6b: %.177 = splice_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %.f75: type = type_of_inst @F.%.loc8_25.13 [template]
// CHECK:STDOUT:   %.82c5: %.f75 = splice_inst @F.%.loc8_25.13 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.fb0a62.1: type = type_of_inst @F.%.loc8_3.17 [template]
// CHECK:STDOUT:   %.42a05f.1: %.fb0a62.1 = splice_inst @F.%.loc8_3.17 [template]
// CHECK:STDOUT:   %.dcfccd.1: type = type_of_inst @F.%.loc8_3.20 [template]
// CHECK:STDOUT:   %.f8b3eb.1: %.dcfccd.1 = splice_inst @F.%.loc8_3.20 [template]
// CHECK:STDOUT:   %.d06: type = type_of_inst @F.%.loc8_3.23 [template]
// CHECK:STDOUT:   %.ebe: %.d06 = splice_inst @F.%.loc8_3.23 [template]
// CHECK:STDOUT:   %.ce0: type = type_of_inst @F.%.loc8_3.26 [template]
// CHECK:STDOUT:   %.9de: %.ce0 = splice_inst @F.%.loc8_3.26 [template]
// CHECK:STDOUT:   %.aee: %.8bd = splice_inst @F.%.loc8_3.29 [template]
// CHECK:STDOUT:   %.d37: %.8bd = splice_inst @F.%.loc8_3.31 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.200: %.8bd = splice_inst @F.%.loc8_3.33 [template]
// CHECK:STDOUT:   %.6a6: type = type_of_inst @F.%.loc8_3.35 [template]
// CHECK:STDOUT:   %.cdc: %.6a6 = splice_inst @F.%.loc8_3.35 [template]
// CHECK:STDOUT:   %.5e7: type = type_of_inst @F.%.loc8_3.38 [template]
// CHECK:STDOUT:   %.993: %.5e7 = splice_inst @F.%.loc8_3.38 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %x.param_patt.468: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.af8: %pattern_type.fa7 = wrapper_binding_pattern x, %x.param_patt.468 [concrete]
// CHECK:STDOUT:   %inst.specific_inst.94c: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fe6: type = specific_inst %X, @F(%X) [concrete = %X]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %inst.splice_block.5af: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.788: type = splice_block imports.%S.decl [concrete = %S] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.c1e: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.895: type = splice_block %S [concrete = %S] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pattern_type.981: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %v.patt.5fa: %pattern_type.981 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.e69: %pattern_type.981 = var_pattern %v.patt.5fa [concrete]
// CHECK:STDOUT:   %.e53: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%S) [concrete]
// CHECK:STDOUT:   %S.Op.type.83af6e.1: type = fn_type @S.Op.1 [concrete]
// CHECK:STDOUT:   %S.Op.4dca31.1: %S.Op.type.83af6e.1 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%S.Op.4dca31.1), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.602: %Default.type = facet_value %S, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.1ff: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.602) [concrete]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.c7b: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.602) [concrete]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.83d: %T.as_type.as.DefaultOrUnformed.impl.Op.type.c7b = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ba3: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.1ff) [concrete]
// CHECK:STDOUT:   %inst.specific_inst.15d: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.3a2: %DefaultOrUnformed.type = specific_inst %DefaultOrUnformed.facet.ba3, @F(%X) [concrete = %DefaultOrUnformed.facet.ba3]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.483: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.ba3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.assoc_type: type = assoc_entity_type @DefaultOrUnformed [concrete]
// CHECK:STDOUT:   %assoc0.b81: %DefaultOrUnformed.assoc_type = assoc_entity element0, imports.%Core.import_ref.d14 [concrete]
// CHECK:STDOUT:   %.805: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.483, %DefaultOrUnformed.facet.ba3 [concrete]
// CHECK:STDOUT:   %inst.splice_block.9d9: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.64e: %.805 = splice_block %impl.elem0.9ae [concrete = %T.as_type.as.DefaultOrUnformed.impl.Op.83d] {
// CHECK:STDOUT:       %as_type: type = facet_access_type %DefaultOrUnformed.facet.ba3 [concrete = %S]
// CHECK:STDOUT:       %.bcb: type = converted %DefaultOrUnformed.facet.ba3, %as_type [concrete = %S]
// CHECK:STDOUT:       %Op.ref.f8c: %DefaultOrUnformed.assoc_type = name_ref Op, imports.%Core.import_ref.6dc [concrete = %assoc0.b81]
// CHECK:STDOUT:       %impl.elem0.9ae: %.805 = impl_witness_access %DefaultOrUnformed.impl_witness.1ff, element0 [concrete = %T.as_type.as.DefaultOrUnformed.impl.Op.83d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.7e5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.cc7: init %S = splice_block %T.as_type.as.DefaultOrUnformed.impl.Op.call {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.746: ref %S = temporary_storage
// CHECK:STDOUT:       %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %S to %.746 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.e6b3f4.2()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.d80: <instruction> = call_action (%ImplicitAs.generic, %.8bd), false [template]
// CHECK:STDOUT:   %.fb0a62.2: type = type_of_inst %.d80 [template]
// CHECK:STDOUT:   %.42a05f.2: %.fb0a62.2 = splice_inst %.d80 [template]
// CHECK:STDOUT:   %.126: <instruction> = access_member_action %.42a05f.2, Convert [template]
// CHECK:STDOUT:   %.dcfccd.2: type = type_of_inst %.126 [template]
// CHECK:STDOUT:   %.f8b3eb.2: %.dcfccd.2 = splice_inst %.126 [template]
// CHECK:STDOUT:   %.609: <instruction> = compound_member_access_action %.cc7, %.f8b3eb.2 [template]
// CHECK:STDOUT:   %.b7f: type = type_of_inst %.609 [template]
// CHECK:STDOUT:   %.c5b: %.b7f = splice_inst %.609 [template]
// CHECK:STDOUT:   %.ee1: <instruction> = call_action (%.c5b), true [template]
// CHECK:STDOUT:   %.fb6: type = type_of_inst %.ee1 [template]
// CHECK:STDOUT:   %.904: %.fb6 = splice_inst %.ee1 [template]
// CHECK:STDOUT:   %.3aa: <instruction> = refine_inst_action %.904 [template]
// CHECK:STDOUT:   %.443: %S = splice_inst %.3aa [template]
// CHECK:STDOUT:   %.00d: <instruction> = convert_to_category_action %.443, element10 [template]
// CHECK:STDOUT:   %.9c0: %S = splice_inst %.00d [template]
// CHECK:STDOUT:   %inst.specific_inst.54a: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.b44: ref %S = specific_inst @F.%v.var, @F(%X)
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.Op.type.83af6e.2: type = fn_type @S.Op.2 [concrete]
// CHECK:STDOUT:   %S.Op.4dca31.2: %S.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.2: <witness> = custom_witness (%S.Op.4dca31.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2e5: %Destroy.type = facet_value %S, (%custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5df: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2e5) [concrete]
// CHECK:STDOUT:   %.670: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5df, %Destroy.facet.2e5 [concrete]
// CHECK:STDOUT:   %inst.splice_block.57b: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.39a: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %impl.elem0.c3e: %.670 = impl_witness_access %custom_witness.e7a31c.2, element0 [concrete = %S.Op.4dca31.2]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.b44, %impl.elem0.c3e
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.8eb: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.a23: init %empty_tuple.type = splice_block %S.cpp_destructor.call {
// CHECK:STDOUT:       %Op.ref.6a0: %S.cpp_destructor.type = name_ref Op, imports.%S.cpp_destructor.decl [concrete = %S.cpp_destructor]
// CHECK:STDOUT:       %S.cpp_destructor.bound: <bound method> = bound_method %.b44, %Op.ref.6a0
// CHECK:STDOUT:       %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%.b44)
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .S = %S.template
// CHECK:STDOUT:     .X = %X.decl
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.template: %S.type = struct_value () [concrete = constants.%S.template]
// CHECK:STDOUT:   %Core.import_ref.6dc: %DefaultOrUnformed.assoc_type = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0.b81]
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.918: @Destroy.WithSelf.%Destroy.WithSelf.Op.type (%Destroy.WithSelf.Op.type.d3e) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.Op (constants.%Destroy.WithSelf.Op.42b)]
// CHECK:STDOUT:   %X.decl: type = class_decl @X [concrete = constants.%X] {} {}
// CHECK:STDOUT:   %S.decl: type = class_decl @S [concrete = constants.%S] {} {}
// CHECK:STDOUT:   %Core.import_ref.d14: @DefaultOrUnformed.WithSelf.%DefaultOrUnformed.WithSelf.Op.type (%DefaultOrUnformed.WithSelf.Op.type.8f1) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @DefaultOrUnformed.WithSelf.%DefaultOrUnformed.WithSelf.Op (constants.%DefaultOrUnformed.WithSelf.Op.ad7)]
// CHECK:STDOUT:   %S.cpp_destructor.decl: %S.cpp_destructor.type = fn_decl @S.cpp_destructor [concrete = constants.%S.cpp_destructor] {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:     %x.param_patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc7_23.1 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: %type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %x.param: @F.%T.loc7_7.1 (%T.67d) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc7: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:     %x: @F.%T.loc7_7.1 (%T.67d) = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type) {
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.98f = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt.d47)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc7_23.2 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc8_23.2: type = template_inst %T.loc7_7.1 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc8_24.6: <instruction> = refine_inst_action %.loc8_23.1 [template]
// CHECK:STDOUT:   %.loc8_24.7: type = splice_inst %.loc8_24.6 [template = %.loc8_24.7 (constants.%.4a6)]
// CHECK:STDOUT:   %.loc8_24.8: <instruction> = call_action (%S.ref, %.loc8_24.2), false [template]
// CHECK:STDOUT:   %.loc8_24.9: type = type_of_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.52a)]
// CHECK:STDOUT:   %.loc8_24.10: @F.%.loc8_24.9 (%.52a) = splice_inst %.loc8_24.8 [template = %.loc8_24.10 (constants.%.2a3)]
// CHECK:STDOUT:   %.loc8_24.11: <instruction> = convert_to_value_action %.loc8_24.4, type [template]
// CHECK:STDOUT:   %.loc8_24.12: type = splice_inst %.loc8_24.11 [template = %.loc8_24.12 (constants.%.8bd)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %.loc8_24.12 [template = %require_complete.loc8 (constants.%require_complete.cfd)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %.loc8_24.12 [template = %pattern_type.loc8 (constants.%pattern_type.0fb)]
// CHECK:STDOUT:   %v.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.0fb) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.5bf)]
// CHECK:STDOUT:   %v.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.0fb) = var_pattern %v.patt.loc8_15.2 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.2d3)]
// CHECK:STDOUT:   %.loc8_25.7: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.loc8_24.12) [template = %.loc8_25.7 (constants.%.892)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.loc8_24.12, @DefaultOrUnformed [template = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2: %DefaultOrUnformed.type = facet_value %.loc8_24.12, (%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.77c)]
// CHECK:STDOUT:   %.loc8_25.8: <instruction> = refine_inst_action %.loc8_25.1 [template]
// CHECK:STDOUT:   %.loc8_25.9: %DefaultOrUnformed.type = splice_inst %.loc8_25.8 [template = %.loc8_25.9 (constants.%.92d)]
// CHECK:STDOUT:   %.loc8_25.10: <instruction> = access_member_action %.loc8_25.2, Op [template]
// CHECK:STDOUT:   %.loc8_25.11: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.177)]
// CHECK:STDOUT:   %.loc8_25.12: @F.%.loc8_25.11 (%.177) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.f6b)]
// CHECK:STDOUT:   %.loc8_25.13: <instruction> = call_action (%.loc8_25.4), false [template]
// CHECK:STDOUT:   %.loc8_25.14: type = type_of_inst %.loc8_25.13 [template = %.loc8_25.14 (constants.%.f75)]
// CHECK:STDOUT:   %.loc8_25.15: @F.%.loc8_25.14 (%.f75) = splice_inst %.loc8_25.13 [template = %.loc8_25.15 (constants.%.82c5)]
// CHECK:STDOUT:   %.loc8_3.17: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.8bd), false [template]
// CHECK:STDOUT:   %.loc8_3.18: type = type_of_inst %.loc8_3.17 [template = %.loc8_3.18 (constants.%.fb0a62.1)]
// CHECK:STDOUT:   %.loc8_3.19: @F.%.loc8_3.18 (%.fb0a62.1) = splice_inst %.loc8_3.17 [template = %.loc8_3.19 (constants.%.42a05f.1)]
// CHECK:STDOUT:   %.loc8_3.20: <instruction> = access_member_action %.loc8_3.2, Convert [template]
// CHECK:STDOUT:   %.loc8_3.21: type = type_of_inst %.loc8_3.20 [template = %.loc8_3.21 (constants.%.dcfccd.1)]
// CHECK:STDOUT:   %.loc8_3.22: @F.%.loc8_3.21 (%.dcfccd.1) = splice_inst %.loc8_3.20 [template = %.loc8_3.22 (constants.%.f8b3eb.1)]
// CHECK:STDOUT:   %.loc8_3.23: <instruction> = compound_member_access_action %.loc8_25.6, %.loc8_3.4 [template]
// CHECK:STDOUT:   %.loc8_3.24: type = type_of_inst %.loc8_3.23 [template = %.loc8_3.24 (constants.%.d06)]
// CHECK:STDOUT:   %.loc8_3.25: @F.%.loc8_3.24 (%.d06) = splice_inst %.loc8_3.23 [template = %.loc8_3.25 (constants.%.ebe)]
// CHECK:STDOUT:   %.loc8_3.26: <instruction> = call_action (%.loc8_3.6), true [template]
// CHECK:STDOUT:   %.loc8_3.27: type = type_of_inst %.loc8_3.26 [template = %.loc8_3.27 (constants.%.ce0)]
// CHECK:STDOUT:   %.loc8_3.28: @F.%.loc8_3.27 (%.ce0) = splice_inst %.loc8_3.26 [template = %.loc8_3.28 (constants.%.9de)]
// CHECK:STDOUT:   %.loc8_3.29: <instruction> = refine_inst_action %.loc8_3.9 [template]
// CHECK:STDOUT:   %.loc8_3.30: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.29 [template = %.loc8_3.30 (constants.%.aee)]
// CHECK:STDOUT:   %.loc8_3.31: <instruction> = convert_to_category_action %.loc8_3.10, element10 [template]
// CHECK:STDOUT:   %.loc8_3.32: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.31 [template = %.loc8_3.32 (constants.%.d37)]
// CHECK:STDOUT:   %.loc8_3.33: <instruction> = refine_inst_action %v.var [template]
// CHECK:STDOUT:   %.loc8_3.34: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.33 [template = %.loc8_3.34 (constants.%.200)]
// CHECK:STDOUT:   %.loc8_3.35: <instruction> = compound_member_access_action %.loc8_3.12, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc8_3.36: type = type_of_inst %.loc8_3.35 [template = %.loc8_3.36 (constants.%.6a6)]
// CHECK:STDOUT:   %.loc8_3.37: @F.%.loc8_3.36 (%.6a6) = splice_inst %.loc8_3.35 [template = %.loc8_3.37 (constants.%.cdc)]
// CHECK:STDOUT:   %.loc8_3.38: <instruction> = call_action (%.loc8_3.14), true [template]
// CHECK:STDOUT:   %.loc8_3.39: type = type_of_inst %.loc8_3.38 [template = %.loc8_3.39 (constants.%.5e7)]
// CHECK:STDOUT:   %.loc8_3.40: @F.%.loc8_3.39 (%.5e7) = splice_inst %.loc8_3.38 [template = %.loc8_3.40 (constants.%.993)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc7_7.1 (%T.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc8_24.12 (%.8bd) = var_storage %v.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_25.1: %DefaultOrUnformed.type = facet_value constants.%.8bd, (constants.%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.77c)]
// CHECK:STDOUT:     %.loc8_25.1: %DefaultOrUnformed.type = converted constants.%.8bd, %DefaultOrUnformed.facet.loc8_25.1 [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.77c)]
// CHECK:STDOUT:     %.loc8_25.2: %DefaultOrUnformed.type = splice_inst %.loc8_25.8 [template = %.loc8_25.9 (constants.%.92d)]
// CHECK:STDOUT:     %.loc8_25.3: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.177)]
// CHECK:STDOUT:     %.loc8_25.4: @F.%.loc8_25.11 (%.177) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.f6b)]
// CHECK:STDOUT:     %.loc8_25.5: type = type_of_inst %.loc8_25.13 [template = %.loc8_25.14 (constants.%.f75)]
// CHECK:STDOUT:     %.loc8_25.6: @F.%.loc8_25.14 (%.f75) = splice_inst %.loc8_25.13 [template = %.loc8_25.15 (constants.%.82c5)]
// CHECK:STDOUT:     %.loc8_3.1: type = type_of_inst %.loc8_3.17 [template = %.loc8_3.18 (constants.%.fb0a62.1)]
// CHECK:STDOUT:     %.loc8_3.2: @F.%.loc8_3.18 (%.fb0a62.1) = splice_inst %.loc8_3.17 [template = %.loc8_3.19 (constants.%.42a05f.1)]
// CHECK:STDOUT:     %.loc8_3.3: type = type_of_inst %.loc8_3.20 [template = %.loc8_3.21 (constants.%.dcfccd.1)]
// CHECK:STDOUT:     %.loc8_3.4: @F.%.loc8_3.21 (%.dcfccd.1) = splice_inst %.loc8_3.20 [template = %.loc8_3.22 (constants.%.f8b3eb.1)]
// CHECK:STDOUT:     %.loc8_3.5: type = type_of_inst %.loc8_3.23 [template = %.loc8_3.24 (constants.%.d06)]
// CHECK:STDOUT:     %.loc8_3.6: @F.%.loc8_3.24 (%.d06) = splice_inst %.loc8_3.23 [template = %.loc8_3.25 (constants.%.ebe)]
// CHECK:STDOUT:     %.loc8_3.7: type = type_of_inst %.loc8_3.26 [template = %.loc8_3.27 (constants.%.ce0)]
// CHECK:STDOUT:     %.loc8_3.8: @F.%.loc8_3.27 (%.ce0) = splice_inst %.loc8_3.26 [template = %.loc8_3.28 (constants.%.9de)]
// CHECK:STDOUT:     %.loc8_3.9: @F.%.loc8_24.12 (%.8bd) = converted %.loc8_25.6, %.loc8_3.8 [template = %.loc8_3.28 (constants.%.9de)]
// CHECK:STDOUT:     %.loc8_3.10: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.29 [template = %.loc8_3.30 (constants.%.aee)]
// CHECK:STDOUT:     %.loc8_3.11: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.31 [template = %.loc8_3.32 (constants.%.d37)]
// CHECK:STDOUT:     assign %v.var, %.loc8_3.11
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.5 [template = %.loc8_24.12 (constants.%.8bd)] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %S.ref: %S.type = name_ref S, imports.%S.template [concrete = constants.%S.template]
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc8_23.1: type = template_inst %T.ref.loc8 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc8_24.2: type = splice_inst %.loc8_24.6 [template = %.loc8_24.7 (constants.%.4a6)]
// CHECK:STDOUT:       %.loc8_24.3: type = type_of_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.52a)]
// CHECK:STDOUT:       %.loc8_24.4: @F.%.loc8_24.9 (%.52a) = splice_inst %.loc8_24.8 [template = %.loc8_24.10 (constants.%.2a3)]
// CHECK:STDOUT:       %.loc8_24.5: type = splice_inst %.loc8_24.11 [template = %.loc8_24.12 (constants.%.8bd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc8_24.12 (%.8bd) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.0fb) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.5bf)]
// CHECK:STDOUT:       %v.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.0fb) = var_pattern %v.patt.loc8_15.1 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.2d3)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_3.12: @F.%.loc8_24.12 (%.8bd) = splice_inst %.loc8_3.33 [template = %.loc8_3.34 (constants.%.200)]
// CHECK:STDOUT:     %.loc8_3.13: type = type_of_inst %.loc8_3.35 [template = %.loc8_3.36 (constants.%.6a6)]
// CHECK:STDOUT:     %.loc8_3.14: @F.%.loc8_3.36 (%.6a6) = splice_inst %.loc8_3.35 [template = %.loc8_3.37 (constants.%.cdc)]
// CHECK:STDOUT:     %.loc8_3.15: type = type_of_inst %.loc8_3.38 [template = %.loc8_3.39 (constants.%.5e7)]
// CHECK:STDOUT:     %.loc8_3.16: @F.%.loc8_3.39 (%.5e7) = splice_inst %.loc8_3.38 [template = %.loc8_3.40 (constants.%.993)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc7 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc7_23.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc7_23.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt.d47
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%X
// CHECK:STDOUT:   %pattern_type.loc7 => constants.%pattern_type.fa7
// CHECK:STDOUT:   %x.param_patt.loc7_23.2 => constants.%x.param_patt.468
// CHECK:STDOUT:   %x.patt.loc7_23.2 => constants.%x.patt.af8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type
// CHECK:STDOUT:   %.loc8_23.2 => constants.%X
// CHECK:STDOUT:   %.loc8_24.6 => constants.%inst.specific_inst.94c
// CHECK:STDOUT:   %.loc8_24.7 => constants.%X
// CHECK:STDOUT:   %.loc8_24.8 => constants.%inst.splice_block.5af
// CHECK:STDOUT:   %.loc8_24.9 => type
// CHECK:STDOUT:   %.loc8_24.10 => constants.%S
// CHECK:STDOUT:   %.loc8_24.11 => constants.%inst.splice_block.c1e
// CHECK:STDOUT:   %.loc8_24.12 => constants.%S
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.981
// CHECK:STDOUT:   %v.patt.loc8_15.2 => constants.%v.patt.5fa
// CHECK:STDOUT:   %v.var_patt.loc8_3.2 => constants.%v.var_patt.e69
// CHECK:STDOUT:   %.loc8_25.7 => constants.%.e53
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.1ff
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2 => constants.%DefaultOrUnformed.facet.ba3
// CHECK:STDOUT:   %.loc8_25.8 => constants.%inst.specific_inst.15d
// CHECK:STDOUT:   %.loc8_25.9 => constants.%DefaultOrUnformed.facet.ba3
// CHECK:STDOUT:   %.loc8_25.10 => constants.%inst.splice_block.9d9
// CHECK:STDOUT:   %.loc8_25.11 => constants.%.805
// CHECK:STDOUT:   %.loc8_25.12 => constants.%T.as_type.as.DefaultOrUnformed.impl.Op.83d
// CHECK:STDOUT:   %.loc8_25.13 => constants.%inst.splice_block.7e5
// CHECK:STDOUT:   %.loc8_25.14 => constants.%S
// CHECK:STDOUT:   %.loc8_25.15 => invalid
// CHECK:STDOUT:   %.loc8_3.17 => constants.%.d80
// CHECK:STDOUT:   %.loc8_3.18 => constants.%.fb0a62.2
// CHECK:STDOUT:   %.loc8_3.19 => constants.%.42a05f.2
// CHECK:STDOUT:   %.loc8_3.20 => constants.%.126
// CHECK:STDOUT:   %.loc8_3.21 => constants.%.dcfccd.2
// CHECK:STDOUT:   %.loc8_3.22 => constants.%.f8b3eb.2
// CHECK:STDOUT:   %.loc8_3.23 => constants.%.609
// CHECK:STDOUT:   %.loc8_3.24 => constants.%.b7f
// CHECK:STDOUT:   %.loc8_3.25 => constants.%.c5b
// CHECK:STDOUT:   %.loc8_3.26 => constants.%.ee1
// CHECK:STDOUT:   %.loc8_3.27 => constants.%.fb6
// CHECK:STDOUT:   %.loc8_3.28 => constants.%.904
// CHECK:STDOUT:   %.loc8_3.29 => constants.%.3aa
// CHECK:STDOUT:   %.loc8_3.30 => constants.%.443
// CHECK:STDOUT:   %.loc8_3.31 => constants.%.00d
// CHECK:STDOUT:   %.loc8_3.32 => constants.%.9c0
// CHECK:STDOUT:   %.loc8_3.33 => constants.%inst.specific_inst.54a
// CHECK:STDOUT:   %.loc8_3.34 => invalid
// CHECK:STDOUT:   %.loc8_3.35 => constants.%inst.splice_block.57b
// CHECK:STDOUT:   %.loc8_3.36 => <bound method>
// CHECK:STDOUT:   %.loc8_3.37 => invalid
// CHECK:STDOUT:   %.loc8_3.38 => constants.%inst.splice_block.8eb
// CHECK:STDOUT:   %.loc8_3.39 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_3.40 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
