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
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T.67d [symbolic]
// CHECK:STDOUT:   %x.param_patt.91d: %pattern_type.51d = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.260: %pattern_type.51d = wrapper_binding_pattern x, %x.param_patt.91d [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T.67d [symbolic]
// CHECK:STDOUT:   %S.type: type = cpp_type_template_type S [concrete]
// CHECK:STDOUT:   %S.template: %S.type = struct_value () [concrete]
// CHECK:STDOUT:   %.a92: type = template_inst %T.67d [template]
// CHECK:STDOUT:   %.d46: type = type_of_inst @F.%.loc8_24.5 [template]
// CHECK:STDOUT:   %.de1: %.d46 = splice_inst @F.%.loc8_24.5 [template]
// CHECK:STDOUT:   %.537: type = splice_inst @F.%.loc8_24.8 [template]
// CHECK:STDOUT:   %require_complete.cb0: <witness> = require_complete_type %.537 [template]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %.537 [template]
// CHECK:STDOUT:   %v.patt.126: %pattern_type.fcb = ref_binding_pattern v [template]
// CHECK:STDOUT:   %v.var_patt.768: %pattern_type.fcb = var_pattern %v.patt.126 [template]
// CHECK:STDOUT:   %DefaultOrUnformed.type: type = facet_type <@DefaultOrUnformed> [concrete]
// CHECK:STDOUT:   %Self.935: %DefaultOrUnformed.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.8f1: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%Self.935) [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.ad7: %DefaultOrUnformed.WithSelf.Op.type.8f1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Default.type: type = facet_type <@Default> [concrete]
// CHECK:STDOUT:   %T.b09: %Default.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%T.b09) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.cfe: %T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9 = struct_value () [symbolic]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.537, @DefaultOrUnformed [template]
// CHECK:STDOUT:   %.0681: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.537) [template]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.67e: %DefaultOrUnformed.type = facet_value %.537, (%DefaultOrUnformed.lookup_impl_witness) [template]
// CHECK:STDOUT:   %.77d: type = type_of_inst @F.%.loc8_25.7 [template]
// CHECK:STDOUT:   %.984: %.77d = splice_inst @F.%.loc8_25.7 [template]
// CHECK:STDOUT:   %.b48: type = type_of_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %.e18: %.b48 = splice_inst @F.%.loc8_25.10 [template]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (<instruction>) [concrete]
// CHECK:STDOUT:   %.5d6: %.537 = splice_inst @F.%tuple.elem0 [template]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Self.0e7: %Destroy.type = symbolic_binding Self, 0 [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.4e5: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Self.0e7) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.4a0: %Destroy.WithSelf.SelfDestruct.type.4e5 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.assoc_type: type = assoc_entity_type @Destroy [concrete]
// CHECK:STDOUT:   %assoc2: %Destroy.assoc_type = assoc_entity element2, imports.%Core.import_ref.71d [concrete]
// CHECK:STDOUT:   %.8cb: type = type_of_inst @F.%.loc8_3.8 [template]
// CHECK:STDOUT:   %.c23: %.8cb = splice_inst @F.%.loc8_3.8 [template]
// CHECK:STDOUT:   %.57c: type = type_of_inst @F.%.loc8_3.11 [template]
// CHECK:STDOUT:   %.e63: %.57c = splice_inst @F.%.loc8_3.11 [template]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %X: type = class_type @X [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.fa7: type = pattern_type %X [concrete]
// CHECK:STDOUT:   %x.param_patt.468: %pattern_type.fa7 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.af8: %pattern_type.fa7 = wrapper_binding_pattern x, %x.param_patt.468 [concrete]
// CHECK:STDOUT:   %S: type = class_type @S [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.1: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.788: type = splice_block imports.%S.decl [concrete = %S] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.2: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.895: type = splice_block %S [concrete = %S] {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %pattern_type.981: type = pattern_type %S [concrete]
// CHECK:STDOUT:   %v.patt.5fa: %pattern_type.981 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:   %v.var_patt.e69: %pattern_type.981 = var_pattern %v.patt.5fa [concrete]
// CHECK:STDOUT:   %.e53: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%S) [concrete]
// CHECK:STDOUT:   %S.Op.type: type = fn_type @S.Op [concrete]
// CHECK:STDOUT:   %S.Op: %S.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.9b4: <witness> = custom_witness (%S.Op), @Default [concrete]
// CHECK:STDOUT:   %Default.facet.602: %Default.type = facet_value %S, (%custom_witness.9b4) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness.1ff: <witness> = impl_witness imports.%DefaultOrUnformed.impl_witness_table.162, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.602) [concrete]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.type.c7b: type = fn_type @T.as_type.as.DefaultOrUnformed.impl.Op, @T.as_type.as.DefaultOrUnformed.impl(%Default.facet.602) [concrete]
// CHECK:STDOUT:   %T.as_type.as.DefaultOrUnformed.impl.Op.83d: %T.as_type.as.DefaultOrUnformed.impl.Op.type.c7b = struct_value () [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.ba3: %DefaultOrUnformed.type = facet_value %S, (%DefaultOrUnformed.impl_witness.1ff) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.WithSelf.Op.type.483: type = fn_type @DefaultOrUnformed.WithSelf.Op, @DefaultOrUnformed.WithSelf(%DefaultOrUnformed.facet.ba3) [concrete]
// CHECK:STDOUT:   %DefaultOrUnformed.assoc_type: type = assoc_entity_type @DefaultOrUnformed [concrete]
// CHECK:STDOUT:   %assoc0: %DefaultOrUnformed.assoc_type = assoc_entity element0, imports.%Core.import_ref.d14 [concrete]
// CHECK:STDOUT:   %.805: type = fn_type_with_self_type %DefaultOrUnformed.WithSelf.Op.type.483, %DefaultOrUnformed.facet.ba3 [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.3: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.64e: %.805 = splice_block %impl.elem0.9ae [concrete = %T.as_type.as.DefaultOrUnformed.impl.Op.83d] {
// CHECK:STDOUT:       %as_type: type = facet_access_type %DefaultOrUnformed.facet.ba3 [concrete = %S]
// CHECK:STDOUT:       %.bcb: type = converted %DefaultOrUnformed.facet.ba3, %as_type [concrete = %S]
// CHECK:STDOUT:       %Op.ref: %DefaultOrUnformed.assoc_type = name_ref Op, imports.%Core.import_ref.6dc [concrete = %assoc0]
// CHECK:STDOUT:       %impl.elem0.9ae: %.805 = impl_witness_access %DefaultOrUnformed.impl_witness.1ff, element0 [concrete = %T.as_type.as.DefaultOrUnformed.impl.Op.83d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.4: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.941: init %S = splice_block %T.as_type.as.DefaultOrUnformed.impl.Op.call {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %.46b016.2: ref %S = specific_inst @F.%v.var, @F(%X)
// CHECK:STDOUT:       %T.as_type.as.DefaultOrUnformed.impl.Op.call: init %S to %.46b016.2 = call %T.as_type.as.DefaultOrUnformed.impl.Op.specific_fn.e6b3f4.2()
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.splice_block.426619.5: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.75f: init %S = splice_block %.941 {}
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%inst.splice_block.426619.5) [concrete]
// CHECK:STDOUT:   %self.param_patt.471: %pattern_type.981 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.5f0: %pattern_type.981 = wrapper_binding_pattern self, %self.param_patt.471 [concrete]
// CHECK:STDOUT:   %S.cpp_destructor.type: type = fn_type @S.cpp_destructor [concrete]
// CHECK:STDOUT:   %S.cpp_destructor: %S.cpp_destructor.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc8 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %S.SubobjectDestroy.type: type = fn_type @S.SubobjectDestroy [concrete]
// CHECK:STDOUT:   %S.SubobjectDestroy: %S.SubobjectDestroy.type = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.1bfa2a.2: <witness> = custom_witness (%Destroy.WithSelf.Op.403171.2, %S.SubobjectDestroy, %Destroy.WithSelf.SelfDestruct.db3fdb.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.edc: %Destroy.type = facet_value %S, (%custom_witness.1bfa2a.2) [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.e44: type = fn_type @Destroy.WithSelf.SelfDestruct.1, @Destroy.WithSelf(%Destroy.facet.edc) [concrete]
// CHECK:STDOUT:   %.bb1: type = fn_type_with_self_type %Destroy.WithSelf.SelfDestruct.type.e44, %Destroy.facet.edc [concrete]
// CHECK:STDOUT:   %inst.splice_block.426619.6: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %.9a4: <bound method> = splice_block %bound_method {
// CHECK:STDOUT:       %.46b016.1: ref %S = specific_inst @F.%v.var, @F(%X)
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %impl.elem2: %.bb1 = impl_witness_access %custom_witness.1bfa2a.2, element2 [concrete = %Destroy.WithSelf.SelfDestruct.db3fdb.2]
// CHECK:STDOUT:       %bound_method: <bound method> = bound_method %.46b016.1, %impl.elem2
// CHECK:STDOUT:     }
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %inst.call: <instruction> = inst_value [concrete] {
// CHECK:STDOUT:     %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call %.9a4(%.46b016.1)
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
// CHECK:STDOUT:   %Core.import_ref.6dc: %DefaultOrUnformed.assoc_type = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [concrete = constants.%assoc0]
// CHECK:STDOUT:   %Core.import_ref.c2f: @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op.type (%T.as_type.as.DefaultOrUnformed.impl.Op.type.0c9) = import_ref Core//prelude/parts/default, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DefaultOrUnformed.impl.%T.as_type.as.DefaultOrUnformed.impl.Op (constants.%T.as_type.as.DefaultOrUnformed.impl.Op.cfe)]
// CHECK:STDOUT:   %DefaultOrUnformed.impl_witness_table.162 = impl_witness_table (%Core.import_ref.c2f), @T.as_type.as.DefaultOrUnformed.impl [concrete]
// CHECK:STDOUT:   %Core.import_ref.71d: @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct.type (%Destroy.WithSelf.SelfDestruct.type.4e5) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @Destroy.WithSelf.%Destroy.WithSelf.SelfDestruct (constants.%Destroy.WithSelf.SelfDestruct.4a0)]
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
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc8 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.981 = ref_param_pattern [concrete = constants.%self.param_patt.471]
// CHECK:STDOUT:     %self.patt: %pattern_type.981 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.5f0]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %S = ref_param call_param0
// CHECK:STDOUT:     %self: ref %S = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt.3a0)]
// CHECK:STDOUT:     %x.param_patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:     %x.patt.loc7_23.1: @F.%pattern_type.loc7 (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc7_23.1 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
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
// CHECK:STDOUT:   %pattern_type.loc7: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %x.param_patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d) = value_param_pattern [symbolic = %x.param_patt.loc7_23.2 (constants.%x.param_patt.91d)]
// CHECK:STDOUT:   %x.patt.loc7_23.2: @F.%pattern_type.loc7 (%pattern_type.51d) = wrapper_binding_pattern x, %x.param_patt.loc7_23.2 [symbolic = %x.patt.loc7_23.2 (constants.%x.patt.260)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7 (constants.%require_complete.944)]
// CHECK:STDOUT:   %.loc8_23.2: type = template_inst %T.loc7_7.1 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:   %.loc8_24.5: <instruction> = call_action (%S.ref, %.loc8_23.1), false [template]
// CHECK:STDOUT:   %.loc8_24.6: type = type_of_inst %.loc8_24.5 [template = %.loc8_24.6 (constants.%.d46)]
// CHECK:STDOUT:   %.loc8_24.7: @F.%.loc8_24.6 (%.d46) = splice_inst %.loc8_24.5 [template = %.loc8_24.7 (constants.%.de1)]
// CHECK:STDOUT:   %.loc8_24.8: <instruction> = convert_to_value_action %.loc8_24.3, type [template]
// CHECK:STDOUT:   %.loc8_24.9: type = splice_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.537)]
// CHECK:STDOUT:   %require_complete.loc8: <witness> = require_complete_type %.loc8_24.9 [template = %require_complete.loc8 (constants.%require_complete.cb0)]
// CHECK:STDOUT:   %pattern_type.loc8: type = pattern_type %.loc8_24.9 [template = %pattern_type.loc8 (constants.%pattern_type.fcb)]
// CHECK:STDOUT:   %v.patt.loc8_15.2: @F.%pattern_type.loc8 (%pattern_type.fcb) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.126)]
// CHECK:STDOUT:   %v.var_patt.loc8_3.2: @F.%pattern_type.loc8 (%pattern_type.fcb) = var_pattern %v.patt.loc8_15.2 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.768)]
// CHECK:STDOUT:   %.loc8_25.6: require_specific_def_type = require_specific_def @T.as.DefaultOrUnformed.impl(%.loc8_24.9) [template = %.loc8_25.6 (constants.%.0681)]
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness: <witness> = lookup_impl_witness %.loc8_24.9, @DefaultOrUnformed [template = %DefaultOrUnformed.lookup_impl_witness (constants.%DefaultOrUnformed.lookup_impl_witness)]
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2: %DefaultOrUnformed.type = facet_value %.loc8_24.9, (%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.67e)]
// CHECK:STDOUT:   %.loc8_25.7: <instruction> = access_member_action %.loc8_25.1, Op [template]
// CHECK:STDOUT:   %.loc8_25.8: type = type_of_inst %.loc8_25.7 [template = %.loc8_25.8 (constants.%.77d)]
// CHECK:STDOUT:   %.loc8_25.9: @F.%.loc8_25.8 (%.77d) = splice_inst %.loc8_25.7 [template = %.loc8_25.9 (constants.%.984)]
// CHECK:STDOUT:   %.loc8_25.10: <instruction> = call_action (%.loc8_25.3), false [template]
// CHECK:STDOUT:   %.loc8_25.11: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.b48)]
// CHECK:STDOUT:   %.loc8_25.12: @F.%.loc8_25.11 (%.b48) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.e18)]
// CHECK:STDOUT:   %.loc8_3.6: %tuple.type = initialize_action %.loc8_25.5, %.loc8_24.9, %v.var, false [template]
// CHECK:STDOUT:   %tuple.elem0: <instruction> = tuple_access %.loc8_3.6, element0 [template]
// CHECK:STDOUT:   %.loc8_3.7: @F.%.loc8_24.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc8_3.7 (constants.%.5d6)]
// CHECK:STDOUT:   %.loc8_3.8: <instruction> = compound_member_access_action %v.var, constants.%assoc2 [template]
// CHECK:STDOUT:   %.loc8_3.9: type = type_of_inst %.loc8_3.8 [template = %.loc8_3.9 (constants.%.8cb)]
// CHECK:STDOUT:   %.loc8_3.10: @F.%.loc8_3.9 (%.8cb) = splice_inst %.loc8_3.8 [template = %.loc8_3.10 (constants.%.c23)]
// CHECK:STDOUT:   %.loc8_3.11: <instruction> = call_action (%.loc8_3.3), true [template]
// CHECK:STDOUT:   %.loc8_3.12: type = type_of_inst %.loc8_3.11 [template = %.loc8_3.12 (constants.%.57c)]
// CHECK:STDOUT:   %.loc8_3.13: @F.%.loc8_3.12 (%.57c) = splice_inst %.loc8_3.11 [template = %.loc8_3.13 (constants.%.e63)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%T.loc7_7.1 (%T.67d)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %v.var: ref @F.%.loc8_24.9 (%.537) = var_storage %v.var_patt.loc8_3.1
// CHECK:STDOUT:     %DefaultOrUnformed.facet.loc8_25.1: %DefaultOrUnformed.type = facet_value constants.%.537, (constants.%DefaultOrUnformed.lookup_impl_witness) [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.67e)]
// CHECK:STDOUT:     %.loc8_25.1: %DefaultOrUnformed.type = converted constants.%.537, %DefaultOrUnformed.facet.loc8_25.1 [template = %DefaultOrUnformed.facet.loc8_25.2 (constants.%DefaultOrUnformed.facet.67e)]
// CHECK:STDOUT:     %.loc8_25.2: type = type_of_inst %.loc8_25.7 [template = %.loc8_25.8 (constants.%.77d)]
// CHECK:STDOUT:     %.loc8_25.3: @F.%.loc8_25.8 (%.77d) = splice_inst %.loc8_25.7 [template = %.loc8_25.9 (constants.%.984)]
// CHECK:STDOUT:     %.loc8_25.4: type = type_of_inst %.loc8_25.10 [template = %.loc8_25.11 (constants.%.b48)]
// CHECK:STDOUT:     %.loc8_25.5: @F.%.loc8_25.11 (%.b48) = splice_inst %.loc8_25.10 [template = %.loc8_25.12 (constants.%.e18)]
// CHECK:STDOUT:     %.loc8_3.1: @F.%.loc8_24.9 (%.537) = splice_inst %tuple.elem0 [template = %.loc8_3.7 (constants.%.5d6)]
// CHECK:STDOUT:     assign %v.var, %.loc8_3.1
// CHECK:STDOUT:     %.loc8_24.1: type = splice_block %.loc8_24.4 [template = %.loc8_24.9 (constants.%.537)] {
// CHECK:STDOUT:       %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:       %S.ref: %S.type = name_ref S, imports.%S.template [concrete = constants.%S.template]
// CHECK:STDOUT:       %T.ref.loc8: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T.67d)]
// CHECK:STDOUT:       %.loc8_23.1: type = template_inst %T.ref.loc8 [template = %.loc8_23.2 (constants.%.a92)]
// CHECK:STDOUT:       %.loc8_24.2: type = type_of_inst %.loc8_24.5 [template = %.loc8_24.6 (constants.%.d46)]
// CHECK:STDOUT:       %.loc8_24.3: @F.%.loc8_24.6 (%.d46) = splice_inst %.loc8_24.5 [template = %.loc8_24.7 (constants.%.de1)]
// CHECK:STDOUT:       %.loc8_24.4: type = splice_inst %.loc8_24.8 [template = %.loc8_24.9 (constants.%.537)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @F.%.loc8_24.9 (%.537) = wrapper_binding v, %v.var
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt.loc8_15.1: @F.%pattern_type.loc8 (%pattern_type.fcb) = ref_binding_pattern v [template = %v.patt.loc8_15.2 (constants.%v.patt.126)]
// CHECK:STDOUT:       %v.var_patt.loc8_3.1: @F.%pattern_type.loc8 (%pattern_type.fcb) = var_pattern %v.patt.loc8_15.1 [template = %v.var_patt.loc8_3.2 (constants.%v.var_patt.768)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc8_3.2: type = type_of_inst %.loc8_3.8 [template = %.loc8_3.9 (constants.%.8cb)]
// CHECK:STDOUT:     %.loc8_3.3: @F.%.loc8_3.9 (%.8cb) = splice_inst %.loc8_3.8 [template = %.loc8_3.10 (constants.%.c23)]
// CHECK:STDOUT:     %.loc8_3.4: type = type_of_inst %.loc8_3.11 [template = %.loc8_3.12 (constants.%.57c)]
// CHECK:STDOUT:     %.loc8_3.5: @F.%.loc8_3.12 (%.57c) = splice_inst %.loc8_3.11 [template = %.loc8_3.13 (constants.%.e63)]
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc8(%self.param: ref %S) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc8(%self.param: ref %S) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %S.cpp_destructor.call: init %empty_tuple.type = call imports.%S.cpp_destructor.decl(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T.67d) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt.3a0
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T.67d
// CHECK:STDOUT:   %pattern_type.loc7 => constants.%pattern_type.51d
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
// CHECK:STDOUT:   %.loc8_24.5 => constants.%inst.splice_block.426619.1
// CHECK:STDOUT:   %.loc8_24.6 => type
// CHECK:STDOUT:   %.loc8_24.7 => constants.%S
// CHECK:STDOUT:   %.loc8_24.8 => constants.%inst.splice_block.426619.2
// CHECK:STDOUT:   %.loc8_24.9 => constants.%S
// CHECK:STDOUT:   %require_complete.loc8 => constants.%complete_type
// CHECK:STDOUT:   %pattern_type.loc8 => constants.%pattern_type.981
// CHECK:STDOUT:   %v.patt.loc8_15.2 => constants.%v.patt.5fa
// CHECK:STDOUT:   %v.var_patt.loc8_3.2 => constants.%v.var_patt.e69
// CHECK:STDOUT:   %.loc8_25.6 => constants.%.e53
// CHECK:STDOUT:   %DefaultOrUnformed.lookup_impl_witness => constants.%DefaultOrUnformed.impl_witness.1ff
// CHECK:STDOUT:   %DefaultOrUnformed.facet.loc8_25.2 => constants.%DefaultOrUnformed.facet.ba3
// CHECK:STDOUT:   %.loc8_25.7 => constants.%inst.splice_block.426619.3
// CHECK:STDOUT:   %.loc8_25.8 => constants.%.805
// CHECK:STDOUT:   %.loc8_25.9 => constants.%T.as_type.as.DefaultOrUnformed.impl.Op.83d
// CHECK:STDOUT:   %.loc8_25.10 => constants.%inst.splice_block.426619.4
// CHECK:STDOUT:   %.loc8_25.11 => constants.%S
// CHECK:STDOUT:   %.loc8_25.12 => invalid
// CHECK:STDOUT:   %.loc8_3.6 => constants.%tuple
// CHECK:STDOUT:   %tuple.elem0 => constants.%inst.splice_block.426619.5
// CHECK:STDOUT:   %.loc8_3.7 => invalid
// CHECK:STDOUT:   %.loc8_3.8 => constants.%inst.splice_block.426619.6
// CHECK:STDOUT:   %.loc8_3.9 => <bound method>
// CHECK:STDOUT:   %.loc8_3.10 => invalid
// CHECK:STDOUT:   %.loc8_3.11 => constants.%inst.call
// CHECK:STDOUT:   %.loc8_3.12 => constants.%empty_tuple.type
// CHECK:STDOUT:   %.loc8_3.13 => invalid
// CHECK:STDOUT: }
// CHECK:STDOUT:
