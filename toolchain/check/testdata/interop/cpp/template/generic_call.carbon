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
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt.3a0: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
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
// CHECK:STDOUT:   %.d46: type = type_of_inst @F.%.loc8_24.5 [template]
// CHECK:STDOUT:   %.de1: %.d46 = splice_inst @F.%.loc8_24.5 [template]
// CHECK:STDOUT:   %.5cd: type = splice_inst @F.%.loc8_24.8 [template]
// CHECK:STDOUT:   %require_complete.363: <witness> = require_complete_type %.5cd [template]
// CHECK:STDOUT:   %pattern_type.a1f: type = pattern_type %.5cd [template]
// CHECK:STDOUT:   %v.patt.019: %pattern_type.a1f = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.d78: %pattern_type.a1f = var_pattern %v.patt.019 [template]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Self.935: %DefaultOrUnformed.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.8f1: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%Self.935) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.ad7: %DefaultOrUnformed.WithSelf.Op.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.5cd, @DefaultOrUnformed [template]
// CHECK:STDOUT:   %.f4d: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.5cd) [template]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.a1a: %DefaultOrUnformed.type = facet_value %.5cd, (%DefaultOrUnformed.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.c72: type = type_of_inst @F.%.loc8_25.7 [template]
// CHECK:STDOUT:   %.0a6: %.c72 = splice_inst @F.%.loc8_25.7 [template]
// CHECK:STDOUT:   %.463: type = type_of_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %.ba3: %.463 = splice_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %.585e1e.1: type = type_of_inst @F.%.loc8_3.15 [template]
// CHECK:STDOUT:   %.b296a8.1: %.585e1e.1 = splice_inst @F.%.loc8_3.15 [template]
// CHECK:STDOUT:   %.e218d7.1: type = type_of_inst @F.%.loc8_3.18 [template]
// CHECK:STDOUT:   %.f58188.1: %.e218d7.1 = splice_inst @F.%.loc8_3.18 [template]
// CHECK:STDOUT:   %.797: type = type_of_inst @F.%.loc8_3.21 [template]
// CHECK:STDOUT:   %.664: %.797 = splice_inst @F.%.loc8_3.21 [template]
// CHECK:STDOUT:   %.9f0: type = type_of_inst @F.%.loc8_3.24 [template]
// CHECK:STDOUT:   %.0f3: %.9f0 = splice_inst @F.%.loc8_3.24 [template]
// CHECK:STDOUT:   %.e77: %.5cd = splice_inst @F.%.loc8_3.27 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d3e: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.42b: %Destroy.WithSelf.Op.type.d3e = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc0.ae8: %Destroy.assoc_type = assoc_entity element0, imports.%Core.import_ref.918 [concrete]
// CHECK:STDOUT:   %.f94: type = type_of_inst @F.%.loc8_3.29 [template]
// CHECK:STDOUT:   %.457: %.f94 = splice_inst @F.%.loc8_3.29 [template]
// CHECK:STDOUT:   %.98a: type = type_of_inst @F.%.loc8_3.32 [template]
// CHECK:STDOUT:   %.610: %.98a = splice_inst @F.%.loc8_3.32 [template]
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
// CHECK:STDOUT:   %.352: <instruction> = call_action (%ImplicitAs.generic, %.5cd), false [template]
// CHECK:STDOUT:   %.585e1e.2: type = type_of_inst %.352 [template]
// CHECK:STDOUT:   %.b296a8.2: %.585e1e.2 = splice_inst %.352 [template]
// CHECK:STDOUT:   %.556: <instruction> = access_member_action %.b296a8.2, Convert [template]
// CHECK:STDOUT:   %.e218d7.2: type = type_of_inst %.556 [template]
// CHECK:STDOUT:   %.f58188.2: %.e218d7.2 = splice_inst %.556 [template]
// CHECK:STDOUT:   %.955: <instruction> = compound_member_access_action %.cc7, %.f58188.2 [template]
// CHECK:STDOUT:   %.5c9: type = type_of_inst %.955 [template]
// CHECK:STDOUT:   %.2bf: %.5c9 = splice_inst %.955 [template]
// CHECK:STDOUT:   %.9a4: <instruction> = call_action (%.2bf), true [template]
// CHECK:STDOUT:   %.94b: type = type_of_inst %.9a4 [template]
// CHECK:STDOUT:   %.a03: %.94b = splice_inst %.9a4 [template]
// CHECK:STDOUT:   %.169: <instruction> = convert_to_category_action %.a03, element10 [template]
// CHECK:STDOUT:   %.217: %S = splice_inst %.169 [template]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %S.Op.type.83af6e.2: type = fn_type @S.Op.2 [concrete]
// CHECK:STDOUT:   %S.Op.4dca31.2: %S.Op.type.83af6e.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.e7a31c.2: <witness> = custom_witness (%S.Op.4dca31.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.2e5: %Destroy.type = facet_value %S, (%custom_witness.e7a31c.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5df: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.2e5) [concrete]
// CHECK:STDOUT:   %.670: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5df, %Destroy.facet.2e5 [concrete]
// CHECK:STDOUT:   %inst.splice_block.b32: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.ef5: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       %.e7d: ref %S = specific_inst @F.%v.var, @F(%X)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %impl.elem0.c3e: %.670 = impl_witness_access %custom_witness.e7a31c.2, element0 [concrete = %S.Op.4dca31.2]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.e7d, %impl.elem0.c3e
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.5c0: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.052: init %empty_tuple.type = splice_block %S.cpp_destructor.call {
// CHECK:STDOUT:       %Op.ref.6a0: %S.cpp_destructor.type = name_ref Op, imports.%S.cpp_destructor.decl [concrete = %S.cpp_destructor]
// CHECK:STDOUT:       %S.cpp_destructor.bound: <bound method> = bound_method %.e7d, %Op.ref.6a0
// CHECK:STDOUT:       %S.cpp_destructor.call: init %empty_tuple.type = call %S.cpp_destructor.bound(%.e7d)
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
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:     %x.param_patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc7_23.1 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
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
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7 (constants.%pattern_type.51d1c4.1)]
// CHECK:STDOUT:   %x.param_patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d1c4.1) = wrapper_binding_pattern x, %x.param_patt.loc7_23.2 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc8_23.2: type = template_inst %T.loc7_7.1 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc8_24.5: <instruction> = call_action (%S.ref, %.loc8_23.1), false [template]
// CHECK:STDOUT:   %.loc8_24.6: type = type_of_inst %.loc8_24.5 [template = %.loc8_24.6 (constants.%.d46)]
// CHECK:STDOUT:   %.loc8_24.7: @F.%.loc8_24.6 (%.d46) = splice_inst %.loc8_24.5 [template = %.loc8_24.7 (constants.%.de1)]
// CHECK:STDOUT:   %.loc8_24.8: <instruction> = convert_to_value_action %.loc8_24.3, type [template]
// CHECK:STDOUT:   %.loc8_24.9: type = splice_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.5cd)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %.loc8_24.9 [template = %require_complete.loc8 (constants.%require_complete.363)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %.loc8_24.9 [template = %pattern_type.loc8 (constants.%pattern_type.a1f)]
// CHECK:STDOUT:   %v.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.a1f) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.019)]
// CHECK:STDOUT:   %v.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.a1f) = var_pattern %v.patt.loc8_15.2 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.d78)]
// CHECK:STDOUT:   %.loc8_25.6: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.loc8_24.9) [template = %.loc8_25.6 (constants.%.f4d)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.loc8_24.9, @DefaultOrUnformed [template = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2: %DefaultOrUnformed.type = facet_value %.loc8_24.9, (%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.a1a)]
// CHECK:STDOUT:   %.loc8_25.7: <instruction> = access_member_action %.loc8_25.1, Op [template]
// CHECK:STDOUT:   %.loc8_25.8: type = type_of_inst %.loc8_25.7 [template = %.loc8_25.8 (constants.%.c72)]
// CHECK:STDOUT:   %.loc8_25.9: @F.%.loc8_25.8 (%.c72) = splice_inst %.loc8_25.7 [template = %.loc8_25.9 (constants.%.0a6)]
// CHECK:STDOUT:   %.loc8_25.10: <instruction> = call_action (%.loc8_25.3), false [template]
// CHECK:STDOUT:   %.loc8_25.11: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.463)]
// CHECK:STDOUT:   %.loc8_25.12: @F.%.loc8_25.11 (%.463) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.ba3)]
// CHECK:STDOUT:   %.loc8_3.15: <instruction> = call_action (constants.%ImplicitAs.generic, constants.%.5cd), false [template]
// CHECK:STDOUT:   %.loc8_3.16: type = type_of_inst %.loc8_3.15 [template = %.loc8_3.16 (constants.%.585e1e.1)]
// CHECK:STDOUT:   %.loc8_3.17: @F.%.loc8_3.16 (%.585e1e.1) = splice_inst %.loc8_3.15 [template = %.loc8_3.17 (constants.%.b296a8.1)]
// CHECK:STDOUT:   %.loc8_3.18: <instruction> = access_member_action %.loc8_3.2, Convert [template]
// CHECK:STDOUT:   %.loc8_3.19: type = type_of_inst %.loc8_3.18 [template = %.loc8_3.19 (constants.%.e218d7.1)]
// CHECK:STDOUT:   %.loc8_3.20: @F.%.loc8_3.19 (%.e218d7.1) = splice_inst %.loc8_3.18 [template = %.loc8_3.20 (constants.%.f58188.1)]
// CHECK:STDOUT:   %.loc8_3.21: <instruction> = compound_member_access_action %.loc8_25.5, %.loc8_3.4 [template]
// CHECK:STDOUT:   %.loc8_3.22: type = type_of_inst %.loc8_3.21 [template = %.loc8_3.22 (constants.%.797)]
// CHECK:STDOUT:   %.loc8_3.23: @F.%.loc8_3.22 (%.797) = splice_inst %.loc8_3.21 [template = %.loc8_3.23 (constants.%.664)]
// CHECK:STDOUT:   %.loc8_3.24: <instruction> = call_action (%.loc8_3.6), true [template]
// CHECK:STDOUT:   %.loc8_3.25: type = type_of_inst %.loc8_3.24 [template = %.loc8_3.25 (constants.%.9f0)]
// CHECK:STDOUT:   %.loc8_3.26: @F.%.loc8_3.25 (%.9f0) = splice_inst %.loc8_3.24 [template = %.loc8_3.26 (constants.%.0f3)]
// CHECK:STDOUT:   %.loc8_3.27: <instruction> = convert_to_category_action %.loc8_3.9, element10 [template]
// CHECK:STDOUT:   %.loc8_3.28: @F.%.loc8_24.9 (%.5cd) = splice_inst %.loc8_3.27 [template = %.loc8_3.28 (constants.%.e77)]
// CHECK:STDOUT:   %.loc8_3.29: <instruction> = compound_member_access_action %v.var, constants.%assoc0.ae8 [template]
// CHECK:STDOUT:   %.loc8_3.30: type = type_of_inst %.loc8_3.29 [template = %.loc8_3.30 (constants.%.f94)]
// CHECK:STDOUT:   %.loc8_3.31: @F.%.loc8_3.30 (%.f94) = splice_inst %.loc8_3.29 [template = %.loc8_3.31 (constants.%.457)]
// CHECK:STDOUT:   %.loc8_3.32: <instruction> = call_action (%.loc8_3.12), true [template]
// CHECK:STDOUT:   %.loc8_3.33: type = type_of_inst %.loc8_3.32 [template = %.loc8_3.33 (constants.%.98a)]
// CHECK:STDOUT:   %.loc8_3.34: @F.%.loc8_3.33 (%.98a) = splice_inst %.loc8_3.32 [template = %.loc8_3.34 (constants.%.610)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc7_7.1 (%T.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc8_24.9 (%.5cd) = var_storage %v.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_25.1: %DefaultOrUnformed.type = facet_value constants.%.5cd, (constants.%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.a1a)]
// CHECK:STDOUT:     %.loc8_25.1: %DefaultOrUnformed.type = converted constants.%.5cd, %DefaultOrUnformed.facet.loc8_25.1 [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.a1a)]
// CHECK:STDOUT:     %.loc8_25.2: type = type_of_inst %.loc8_25.7 [template = %.loc8_25.8 (constants.%.c72)]
// CHECK:STDOUT:     %.loc8_25.3: @F.%.loc8_25.8 (%.c72) = splice_inst %.loc8_25.7 [template = %.loc8_25.9 (constants.%.0a6)]
// CHECK:STDOUT:     %.loc8_25.4: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.463)]
// CHECK:STDOUT:     %.loc8_25.5: @F.%.loc8_25.11 (%.463) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.ba3)]
// CHECK:STDOUT:     %.loc8_3.1: type = type_of_inst %.loc8_3.15 [template = %.loc8_3.16 (constants.%.585e1e.1)]
// CHECK:STDOUT:     %.loc8_3.2: @F.%.loc8_3.16 (%.585e1e.1) = splice_inst %.loc8_3.15 [template = %.loc8_3.17 (constants.%.b296a8.1)]
// CHECK:STDOUT:     %.loc8_3.3: type = type_of_inst %.loc8_3.18 [template = %.loc8_3.19 (constants.%.e218d7.1)]
// CHECK:STDOUT:     %.loc8_3.4: @F.%.loc8_3.19 (%.e218d7.1) = splice_inst %.loc8_3.18 [template = %.loc8_3.20 (constants.%.f58188.1)]
// CHECK:STDOUT:     %.loc8_3.5: type = type_of_inst %.loc8_3.21 [template = %.loc8_3.22 (constants.%.797)]
// CHECK:STDOUT:     %.loc8_3.6: @F.%.loc8_3.22 (%.797) = splice_inst %.loc8_3.21 [template = %.loc8_3.23 (constants.%.664)]
// CHECK:STDOUT:     %.loc8_3.7: type = type_of_inst %.loc8_3.24 [template = %.loc8_3.25 (constants.%.9f0)]
// CHECK:STDOUT:     %.loc8_3.8: @F.%.loc8_3.25 (%.9f0) = splice_inst %.loc8_3.24 [template = %.loc8_3.26 (constants.%.0f3)]
// CHECK:STDOUT:     %.loc8_3.9: @F.%.loc8_24.9 (%.5cd) = converted %.loc8_25.5, %.loc8_3.8 [template = %.loc8_3.26 (constants.%.0f3)]
// CHECK:STDOUT:     %.loc8_3.10: @F.%.loc8_24.9 (%.5cd) = splice_inst %.loc8_3.27 [template = %.loc8_3.28 (constants.%.e77)]
// CHECK:STDOUT:     assign %v.var, %.loc8_3.10
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.4 [template = %.loc8_24.9 (constants.%.5cd)] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %S.ref: %S.type = name_ref S, imports.%S.template [concrete = constants.%S.template]
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc8_23.1: type = template_inst %T.ref.loc8 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc8_24.2: type = type_of_inst %.loc8_24.5 [template = %.loc8_24.6 (constants.%.d46)]
// CHECK:STDOUT:       %.loc8_24.3: @F.%.loc8_24.6 (%.d46) = splice_inst %.loc8_24.5 [template = %.loc8_24.7 (constants.%.de1)]
// CHECK:STDOUT:       %.loc8_24.4: type = splice_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.5cd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc8_24.9 (%.5cd) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.a1f) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.019)]
// CHECK:STDOUT:       %v.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.a1f) = var_pattern %v.patt.loc8_15.1 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.d78)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_3.11: type = type_of_inst %.loc8_3.29 [template = %.loc8_3.30 (constants.%.f94)]
// CHECK:STDOUT:     %.loc8_3.12: @F.%.loc8_3.30 (%.f94) = splice_inst %.loc8_3.29 [template = %.loc8_3.31 (constants.%.457)]
// CHECK:STDOUT:     %.loc8_3.13: type = type_of_inst %.loc8_3.32 [template = %.loc8_3.33 (constants.%.98a)]
// CHECK:STDOUT:     %.loc8_3.14: @F.%.loc8_3.33 (%.98a) = splice_inst %.loc8_3.32 [template = %.loc8_3.34 (constants.%.610)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc7 => constants.%pattern_type.51d1c4.1
// CHECK:STDOUT:   %x.param_patt.loc7_23.2 => constants.%x.param_patt.91d
// CHECK:STDOUT:   %x.patt.loc7_23.2 => constants.%x.patt.260
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%X) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%X
// CHECK:STDOUT:   %pattern_type.loc7 => constants.%pattern_type.fa7
// CHECK:STDOUT:   %x.param_patt.loc7_23.2 => constants.%x.param_patt.468
// CHECK:STDOUT:   %x.patt.loc7_23.2 => constants.%x.patt.af8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7 => constants.%complete_type
// CHECK:STDOUT:   %.loc8_23.2 => constants.%X
// CHECK:STDOUT:   %.loc8_24.5 => constants.%inst.splice_block.5af
// CHECK:STDOUT:   %.loc8_24.6 => type
// CHECK:STDOUT:   %.loc8_24.7 => constants.%S
// CHECK:STDOUT:   %.loc8_24.8 => constants.%inst.splice_block.c1e
// CHECK:STDOUT:   %.loc8_24.9 => constants.%S
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.981
// CHECK:STDOUT:   %v.patt.loc8_15.2 => constants.%v.patt.5fa
// CHECK:STDOUT:   %v.var_patt.loc8_3.2 => constants.%v.var_patt.e69
// CHECK:STDOUT:   %.loc8_25.6 => constants.%.e53
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.1ff
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2 => constants.%DefaultOrUnformed.facet.ba3
// CHECK:STDOUT:   %.loc8_25.7 => constants.%inst.splice_block.9d9
// CHECK:STDOUT:   %.loc8_25.8 => constants.%.805
// CHECK:STDOUT:   %.loc8_25.9 => constants.%T.as_type.as.DefaultOrUnformed.impl.Op.83d
// CHECK:STDOUT:   %.loc8_25.10 => constants.%inst.splice_block.7e5
// CHECK:STDOUT:   %.loc8_25.11 => constants.%S
// CHECK:STDOUT:   %.loc8_25.12 => invalid
// CHECK:STDOUT:   %.loc8_3.15 => constants.%.352
// CHECK:STDOUT:   %.loc8_3.16 => constants.%.585e1e.2
// CHECK:STDOUT:   %.loc8_3.17 => constants.%.b296a8.2
// CHECK:STDOUT:   %.loc8_3.18 => constants.%.556
// CHECK:STDOUT:   %.loc8_3.19 => constants.%.e218d7.2
// CHECK:STDOUT:   %.loc8_3.20 => constants.%.f58188.2
// CHECK:STDOUT:   %.loc8_3.21 => constants.%.955
// CHECK:STDOUT:   %.loc8_3.22 => constants.%.5c9
// CHECK:STDOUT:   %.loc8_3.23 => constants.%.2bf
// CHECK:STDOUT:   %.loc8_3.24 => constants.%.9a4
// CHECK:STDOUT:   %.loc8_3.25 => constants.%.94b
// CHECK:STDOUT:   %.loc8_3.26 => constants.%.a03
// CHECK:STDOUT:   %.loc8_3.27 => constants.%.169
// CHECK:STDOUT:   %.loc8_3.28 => constants.%.217
// CHECK:STDOUT:   %.loc8_3.29 => constants.%inst.splice_block.b32
// CHECK:STDOUT:   %.loc8_3.30 => <bound method>
// CHECK:STDOUT:   %.loc8_3.31 => invalid
// CHECK:STDOUT:   %.loc8_3.32 => constants.%inst.splice_block.5c0
// CHECK:STDOUT:   %.loc8_3.33 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_3.34 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
