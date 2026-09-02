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
// CHECK:STDOUT:   %.ae6: @F.%.loc8_24.5 (@F.%.loc8_24.5) = splice_inst @F.%.loc8_24.4 [template]
// CHECK:STDOUT:   %.535: type = splice_inst @F.%.loc8_24.7 [template]
// CHECK:STDOUT:   %require_complete.01e: <witness> = require_complete_type %.535 [template]
// CHECK:STDOUT:   %pattern_type.281: type = pattern_type %.535 [template]
// CHECK:STDOUT:   %v.patt.1d5: %pattern_type.281 = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.367: %pattern_type.281 = var_pattern %v.patt.1d5 [template]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Self.935: %DefaultOrUnformed.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.8f1: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%Self.935) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.ad7: %DefaultOrUnformed.WithSelf.Op.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.535, @DefaultOrUnformed [template]
// CHECK:STDOUT:   %.6b4: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.535) [template]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.45f: %DefaultOrUnformed.type = facet_value %.535, (%DefaultOrUnformed.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.bb0: @F.%.loc8_25.6 (@F.%.loc8_25.6) = splice_inst @F.%.loc8_25.5 [template]
// CHECK:STDOUT:   %.cde: @F.%.loc8_25.9 (@F.%.loc8_25.9) = splice_inst @F.%.loc8_25.8 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.847f2c.1: @F.%.loc8_3.12 (@F.%.loc8_3.12) = splice_inst @F.%.loc8_3.11 [template]
// CHECK:STDOUT:   %.7fa0c8.1: @F.%.loc8_3.15 (@F.%.loc8_3.15) = splice_inst @F.%.loc8_3.14 [template]
// CHECK:STDOUT:   %.178: @F.%.loc8_3.18 (@F.%.loc8_3.18) = splice_inst @F.%.loc8_3.17 [template]
// CHECK:STDOUT:   %.852: @F.%.loc8_3.21 (@F.%.loc8_3.21) = splice_inst @F.%.loc8_3.20 [template]
// CHECK:STDOUT:   %.7f5: %.535 = splice_inst @F.%.loc8_3.23 [template]
// CHECK:STDOUT:   %.f22: %.535 = splice_inst @F.%.loc8_3.25 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.131: %.535 = splice_inst @F.%.loc8_3.27 [template]
// CHECK:STDOUT:   %.8f8: @F.%.loc8_3.30 (@F.%.loc8_3.30) = splice_inst @F.%.loc8_3.29 [template]
// CHECK:STDOUT:   %.763: @F.%.loc8_3.33 (@F.%.loc8_3.33) = splice_inst @F.%.loc8_3.32 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %x.param_patt.468: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.af8: %pattern_type.fa7 = wrapper_binding_pattern x, %x.param_patt.468 [concrete]
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
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.483: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.ba3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.assoc_type: type = assoc_entity_type @DefaultOrUnformed [concrete]
// CHECK:STDOUT:   %assoc0.b81: %DefaultOrUnformed.assoc_type = assoc_entity element0, imports.%Core.import_ref.d14 [concrete]
// CHECK:STDOUT:   %.805: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.483, %DefaultOrUnformed.facet.ba3 [concrete]
// CHECK:STDOUT:   %inst.splice_block.9d9: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.64e: %.805 = splice_block %impl.elem0.9ae [concrete = %T.as_type.as.DefaultOrUnformed.impl.Op.83d] {
// CHECK:STDOUT:       %as_type: type = facet_access_type %DefaultOrUnformed.facet.ba3 [concrete = %S]
// CHECK:STDOUT:       %.bcbe: type = converted %DefaultOrUnformed.facet.ba3, %as_type [concrete = %S]
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
// CHECK:STDOUT:   %.04f: <instruction> = call_action (%ImplicitAs.generic, %.535), false [template]
// CHECK:STDOUT:   %.5a1a: type = type_of_inst %.04f [template]
// CHECK:STDOUT:   %.847f2c.2: %.5a1a = splice_inst %.04f [template]
// CHECK:STDOUT:   %.5d9: <instruction> = access_member_action %.847f2c.2, Convert [template]
// CHECK:STDOUT:   %.831: type = type_of_inst %.5d9 [template]
// CHECK:STDOUT:   %.7fa0c8.2: %.831 = splice_inst %.5d9 [template]
// CHECK:STDOUT:   %.fba: <instruction> = compound_member_access_action %.cc7, %.7fa0c8.2 [template]
// CHECK:STDOUT:   %.5a14: type = type_of_inst %.fba [template]
// CHECK:STDOUT:   %.cb9: %.5a14 = splice_inst %.fba [template]
// CHECK:STDOUT:   %.c0e: <instruction> = call_action (%.cb9), true [template]
// CHECK:STDOUT:   %.e96: type = type_of_inst %.c0e [template]
// CHECK:STDOUT:   %.296: %.e96 = splice_inst %.c0e [template]
// CHECK:STDOUT:   %.dd8: %S = splice_inst %.c0e [template]
// CHECK:STDOUT:   %inst.as_compatible.276: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.dfb: %S = as_compatible %.296 [template = %.dd8]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.23d: <instruction> = convert_to_category_action %.dd8, element10 [template]
// CHECK:STDOUT:   %.a98: %S = splice_inst %.23d [template]
// CHECK:STDOUT:   %inst.as_compatible.c08: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.37f: ref %S = as_compatible @F.%v.var
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.Op.type.83af6e.2: type = fn_type @S.Op.2 [concrete]
// CHECK:STDOUT:   %S.Op.4dca31.2: %S.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.2: <witness> = custom_witness (%S.Op.4dca31.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2e5: %Destroy.type = facet_value %S, (%custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5df: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2e5) [concrete]
// CHECK:STDOUT:   %.670: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5df, %Destroy.facet.2e5 [concrete]
// CHECK:STDOUT:   %inst.splice_block.603: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.fe0: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %impl.elem0.c3e: %.670 = impl_witness_access %custom_witness.e7a31c.2, element0 [concrete = %S.Op.4dca31.2]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.37f, %impl.elem0.c3e
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.35a: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.f30: init %empty_tuple.type = splice_block %S.cpp_destructor.call {
// CHECK:STDOUT:       %Op.ref.6a0: %S.cpp_destructor.type = name_ref Op, imports.%S.cpp_destructor.decl [concrete = %S.cpp_destructor]
// CHECK:STDOUT:       %S.cpp_destructor.bound: <bound method> = bound_method %.37f, %Op.ref.6a0
// CHECK:STDOUT:       %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%.37f)
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
// CHECK:STDOUT:   %.loc8_24.4: <instruction> = call_template_action clang_decl_id78000002, (%T.loc7_7.1) [template]
// CHECK:STDOUT:   %.loc8_24.5: type = type_of_inst %.loc8_24.4 [template]
// CHECK:STDOUT:   %.loc8_24.6: @F.%.loc8_24.5 (@F.%.loc8_24.5) = splice_inst %.loc8_24.4 [template = %.loc8_24.6 (constants.%.ae6)]
// CHECK:STDOUT:   %.loc8_24.7: <instruction> = convert_to_value_action %.loc8_24.2, type [template]
// CHECK:STDOUT:   %.loc8_24.8: type = splice_inst %.loc8_24.7 [template = %.loc8_24.8 (constants.%.535)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %.loc8_24.8 [template = %require_complete.loc8 (constants.%require_complete.01e)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %.loc8_24.8 [template = %pattern_type.loc8 (constants.%pattern_type.281)]
// CHECK:STDOUT:   %v.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.281) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.1d5)]
// CHECK:STDOUT:   %v.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.281) = var_pattern %v.patt.loc8_15.2 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.367)]
// CHECK:STDOUT:   %.loc8_25.4: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.loc8_24.8) [template = %.loc8_25.4 (constants.%.6b4)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.loc8_24.8, @DefaultOrUnformed [template = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2: %DefaultOrUnformed.type = facet_value %.loc8_24.8, (%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.45f)]
// CHECK:STDOUT:   %.loc8_25.5: <instruction> = access_member_action %.loc8_25.1, Op [template]
// CHECK:STDOUT:   %.loc8_25.6: type = type_of_inst %.loc8_25.5 [template]
// CHECK:STDOUT:   %.loc8_25.7: @F.%.loc8_25.6 (@F.%.loc8_25.6) = splice_inst %.loc8_25.5 [template = %.loc8_25.7 (constants.%.bb0)]
// CHECK:STDOUT:   %.loc8_25.8: <instruction> = call_action (%.loc8_25.2), false [template]
// CHECK:STDOUT:   %.loc8_25.9: type = type_of_inst %.loc8_25.8 [template]
// CHECK:STDOUT:   %.loc8_25.10: @F.%.loc8_25.9 (@F.%.loc8_25.9) = splice_inst %.loc8_25.8 [template = %.loc8_25.10 (constants.%.cde)]
// CHECK:STDOUT:   %.loc8_3.11: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.535), false [template]
// CHECK:STDOUT:   %.loc8_3.12: type = type_of_inst %.loc8_3.11 [template]
// CHECK:STDOUT:   %.loc8_3.13: @F.%.loc8_3.12 (@F.%.loc8_3.12) = splice_inst %.loc8_3.11 [template = %.loc8_3.13 (constants.%.847f2c.1)]
// CHECK:STDOUT:   %.loc8_3.14: <instruction> = access_member_action %.loc8_3.1, Convert [template]
// CHECK:STDOUT:   %.loc8_3.15: type = type_of_inst %.loc8_3.14 [template]
// CHECK:STDOUT:   %.loc8_3.16: @F.%.loc8_3.15 (@F.%.loc8_3.15) = splice_inst %.loc8_3.14 [template = %.loc8_3.16 (constants.%.7fa0c8.1)]
// CHECK:STDOUT:   %.loc8_3.17: <instruction> = compound_member_access_action %.loc8_25.3, %.loc8_3.2 [template]
// CHECK:STDOUT:   %.loc8_3.18: type = type_of_inst %.loc8_3.17 [template]
// CHECK:STDOUT:   %.loc8_3.19: @F.%.loc8_3.18 (@F.%.loc8_3.18) = splice_inst %.loc8_3.17 [template = %.loc8_3.19 (constants.%.178)]
// CHECK:STDOUT:   %.loc8_3.20: <instruction> = call_action (%.loc8_3.3), true [template]
// CHECK:STDOUT:   %.loc8_3.21: type = type_of_inst %.loc8_3.20 [template]
// CHECK:STDOUT:   %.loc8_3.22: @F.%.loc8_3.21 (@F.%.loc8_3.21) = splice_inst %.loc8_3.20 [template = %.loc8_3.22 (constants.%.852)]
// CHECK:STDOUT:   %.loc8_3.23: <instruction> = refine_type_action %.loc8_3.5, %.loc8_24.8 [template]
// CHECK:STDOUT:   %.loc8_3.24: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.23 [template = %.loc8_3.24 (constants.%.7f5)]
// CHECK:STDOUT:   %.loc8_3.25: <instruction> = convert_to_category_action %.loc8_3.6, element10 [template]
// CHECK:STDOUT:   %.loc8_3.26: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.25 [template = %.loc8_3.26 (constants.%.f22)]
// CHECK:STDOUT:   %.loc8_3.27: <instruction> = refine_type_action %v.var, %.loc8_24.8 [template]
// CHECK:STDOUT:   %.loc8_3.28: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.27 [template = %.loc8_3.28 (constants.%.131)]
// CHECK:STDOUT:   %.loc8_3.29: <instruction> = compound_member_access_action %.loc8_3.8, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc8_3.30: type = type_of_inst %.loc8_3.29 [template]
// CHECK:STDOUT:   %.loc8_3.31: @F.%.loc8_3.30 (@F.%.loc8_3.30) = splice_inst %.loc8_3.29 [template = %.loc8_3.31 (constants.%.8f8)]
// CHECK:STDOUT:   %.loc8_3.32: <instruction> = call_action (%.loc8_3.9), true [template]
// CHECK:STDOUT:   %.loc8_3.33: type = type_of_inst %.loc8_3.32 [template]
// CHECK:STDOUT:   %.loc8_3.34: @F.%.loc8_3.33 (@F.%.loc8_3.33) = splice_inst %.loc8_3.32 [template = %.loc8_3.34 (constants.%.763)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc7_7.1 (%T.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc8_24.8 (%.535) = var_storage %v.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_25.1: %DefaultOrUnformed.type = facet_value constants.%.535, (constants.%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.45f)]
// CHECK:STDOUT:     %.loc8_25.1: %DefaultOrUnformed.type = converted constants.%.535, %DefaultOrUnformed.facet.loc8_25.1 [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.45f)]
// CHECK:STDOUT:     %.loc8_25.2: @F.%.loc8_25.6 (@F.%.loc8_25.6) = splice_inst %.loc8_25.5 [template = %.loc8_25.7 (constants.%.bb0)]
// CHECK:STDOUT:     %.loc8_25.3: @F.%.loc8_25.9 (@F.%.loc8_25.9) = splice_inst %.loc8_25.8 [template = %.loc8_25.10 (constants.%.cde)]
// CHECK:STDOUT:     %.loc8_3.1: @F.%.loc8_3.12 (@F.%.loc8_3.12) = splice_inst %.loc8_3.11 [template = %.loc8_3.13 (constants.%.847f2c.1)]
// CHECK:STDOUT:     %.loc8_3.2: @F.%.loc8_3.15 (@F.%.loc8_3.15) = splice_inst %.loc8_3.14 [template = %.loc8_3.16 (constants.%.7fa0c8.1)]
// CHECK:STDOUT:     %.loc8_3.3: @F.%.loc8_3.18 (@F.%.loc8_3.18) = splice_inst %.loc8_3.17 [template = %.loc8_3.19 (constants.%.178)]
// CHECK:STDOUT:     %.loc8_3.4: @F.%.loc8_3.21 (@F.%.loc8_3.21) = splice_inst %.loc8_3.20 [template = %.loc8_3.22 (constants.%.852)]
// CHECK:STDOUT:     %.loc8_3.5: @F.%.loc8_24.8 (%.535) = converted %.loc8_25.3, %.loc8_3.4 [template = %.loc8_3.22 (constants.%.852)]
// CHECK:STDOUT:     %.loc8_3.6: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.23 [template = %.loc8_3.24 (constants.%.7f5)]
// CHECK:STDOUT:     %.loc8_3.7: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.25 [template = %.loc8_3.26 (constants.%.f22)]
// CHECK:STDOUT:     assign %v.var, %.loc8_3.7
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.3 [template = %.loc8_24.8 (constants.%.535)] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %S.ref: %S.type = name_ref S, imports.%S.template [concrete = constants.%S.template]
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc8_24.2: @F.%.loc8_24.5 (@F.%.loc8_24.5) = splice_inst %.loc8_24.4 [template = %.loc8_24.6 (constants.%.ae6)]
// CHECK:STDOUT:       %.loc8_24.3: type = splice_inst %.loc8_24.7 [template = %.loc8_24.8 (constants.%.535)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc8_24.8 (%.535) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.281) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.1d5)]
// CHECK:STDOUT:       %v.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.281) = var_pattern %v.patt.loc8_15.1 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.367)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_3.8: @F.%.loc8_24.8 (%.535) = splice_inst %.loc8_3.27 [template = %.loc8_3.28 (constants.%.131)]
// CHECK:STDOUT:     %.loc8_3.9: @F.%.loc8_3.30 (@F.%.loc8_3.30) = splice_inst %.loc8_3.29 [template = %.loc8_3.31 (constants.%.8f8)]
// CHECK:STDOUT:     %.loc8_3.10: @F.%.loc8_3.33 (@F.%.loc8_3.33) = splice_inst %.loc8_3.32 [template = %.loc8_3.34 (constants.%.763)]
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
// CHECK:STDOUT:   %.loc8_24.4 => constants.%inst.splice_block.5af
// CHECK:STDOUT:   %.loc8_24.5 => type
// CHECK:STDOUT:   %.loc8_24.6 => constants.%S
// CHECK:STDOUT:   %.loc8_24.7 => constants.%inst.splice_block.c1e
// CHECK:STDOUT:   %.loc8_24.8 => constants.%S
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.981
// CHECK:STDOUT:   %v.patt.loc8_15.2 => constants.%v.patt.5fa
// CHECK:STDOUT:   %v.var_patt.loc8_3.2 => constants.%v.var_patt.e69
// CHECK:STDOUT:   %.loc8_25.4 => constants.%.e53
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.1ff
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2 => constants.%DefaultOrUnformed.facet.ba3
// CHECK:STDOUT:   %.loc8_25.5 => constants.%inst.splice_block.9d9
// CHECK:STDOUT:   %.loc8_25.6 => constants.%.805
// CHECK:STDOUT:   %.loc8_25.7 => constants.%T.as_type.as.DefaultOrUnformed.impl.Op.83d
// CHECK:STDOUT:   %.loc8_25.8 => constants.%inst.splice_block.7e5
// CHECK:STDOUT:   %.loc8_25.9 => constants.%S
// CHECK:STDOUT:   %.loc8_25.10 => invalid
// CHECK:STDOUT:   %.loc8_3.11 => constants.%.04f
// CHECK:STDOUT:   %.loc8_3.12 => constants.%.5a1a
// CHECK:STDOUT:   %.loc8_3.13 => constants.%.847f2c.2
// CHECK:STDOUT:   %.loc8_3.14 => constants.%.5d9
// CHECK:STDOUT:   %.loc8_3.15 => constants.%.831
// CHECK:STDOUT:   %.loc8_3.16 => constants.%.7fa0c8.2
// CHECK:STDOUT:   %.loc8_3.17 => constants.%.fba
// CHECK:STDOUT:   %.loc8_3.18 => constants.%.5a14
// CHECK:STDOUT:   %.loc8_3.19 => constants.%.cb9
// CHECK:STDOUT:   %.loc8_3.20 => constants.%.c0e
// CHECK:STDOUT:   %.loc8_3.21 => constants.%.e96
// CHECK:STDOUT:   %.loc8_3.22 => constants.%.296
// CHECK:STDOUT:   %.loc8_3.23 => constants.%inst.as_compatible.276
// CHECK:STDOUT:   %.loc8_3.24 => constants.%.dd8
// CHECK:STDOUT:   %.loc8_3.25 => constants.%.23d
// CHECK:STDOUT:   %.loc8_3.26 => constants.%.a98
// CHECK:STDOUT:   %.loc8_3.27 => constants.%inst.as_compatible.c08
// CHECK:STDOUT:   %.loc8_3.28 => invalid
// CHECK:STDOUT:   %.loc8_3.29 => constants.%inst.splice_block.603
// CHECK:STDOUT:   %.loc8_3.30 => <bound method>
// CHECK:STDOUT:   %.loc8_3.31 => invalid
// CHECK:STDOUT:   %.loc8_3.32 => constants.%inst.splice_block.35a
// CHECK:STDOUT:   %.loc8_3.33 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_3.34 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
