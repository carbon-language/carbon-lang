// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/const/import.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/const/import.carbon

// --- implicit.carbon
library "[[@TEST_NAME]]";

class C {}

fn F() -> const C;

var a_ref: const C = F();
var a_ptr_ref: const C* = &a_ref;

// --- implicit.impl.carbon
impl library "[[@TEST_NAME]]";

// Take a reference to avoid unsupported copy logic. This still validates the
// `const` is handled.
//@dump-sem-ir-begin
var a: const C* = &a_ref;
var a_ptr: const C* = a_ptr_ref;
//@dump-sem-ir-end

// CHECK:STDOUT: --- implicit.impl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.48f: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.970: %ptr.as.Copy.impl.Op.type.48f = struct_value () [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.0e5: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.c45: type = ptr_type %const.0e5 [concrete]
// CHECK:STDOUT:   %pattern_type.6eb: type = pattern_type %ptr.c45 [concrete]
// CHECK:STDOUT:   %pattern_type.03b: type = pattern_type %const.0e5 [concrete]
// CHECK:STDOUT:   %addr: %ptr.c45 = addr_of imports.%a_ref.var [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.b9c: <witness> = impl_witness imports.%Copy.impl_witness_table.1ed, @ptr.as.Copy.impl(%const.0e5) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.8d3: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.0e5) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.09c: %ptr.as.Copy.impl.Op.type.8d3 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.c45, (%Copy.impl_witness.b9c) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.6aa: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.090: type = fn_type_with_self_type %Copy.WithSelf.Op.type.6aa, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.09c [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.09c, @ptr.as.Copy.impl.Op(%const.0e5) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//implicit, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.a_ref: ref %const.0e5 = import_ref Main//implicit, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Main.a_ptr_ref: ref %ptr.c45 = import_ref Main//implicit, a_ptr_ref, loaded [concrete = %a_ptr_ref.var]
// CHECK:STDOUT:   %Main.import_ref.1cc: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.48f) = import_ref Main//implicit, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.970)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ed = impl_witness_table (%Main.import_ref.1cc), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.03b = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.03b = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %const.0e5 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %a_ptr_ref.patt: %pattern_type.6eb = ref_binding_pattern a_ptr_ref [concrete]
// CHECK:STDOUT:   %a_ptr_ref.var_patt: %pattern_type.6eb = var_pattern %a_ptr_ref.patt [concrete]
// CHECK:STDOUT:   %a_ptr_ref.var: ref %ptr.c45 = var %a_ptr_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.6eb = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.6eb = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr.c45 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %ptr.loc6 [concrete = constants.%ptr.c45] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6: type = const_type %C.ref.loc6 [concrete = constants.%const.0e5]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %const.loc6 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr.c45 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr.patt: %pattern_type.6eb = ref_binding_pattern a_ptr [concrete]
// CHECK:STDOUT:     %a_ptr.var_patt: %pattern_type.6eb = var_pattern %a_ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr.var: ref %ptr.c45 = var %a_ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.c45] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc7: type = const_type %C.ref.loc7 [concrete = constants.%const.0e5]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %const.loc7 [concrete = constants.%ptr.c45]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr: ref %ptr.c45 = ref_binding a_ptr, %a_ptr.var [concrete = %a_ptr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %const.0e5 = name_ref a_ref, imports.%Main.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %addr: %ptr.c45 = addr_of %a_ref.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0.loc6: %.090 = impl_witness_access constants.%Copy.impl_witness.b9c, element0 [concrete = constants.%ptr.as.Copy.impl.Op.09c]
// CHECK:STDOUT:   %bound_method.loc6_19.1: <bound method> = bound_method %addr, %impl.elem0.loc6 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @ptr.as.Copy.impl.Op(constants.%const.0e5) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_19.2: <bound method> = bound_method %addr, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc6: init %ptr.c45 = call %bound_method.loc6_19.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%a.var, %ptr.as.Copy.impl.Op.call.loc6
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %ptr.c45 = name_ref a_ptr_ref, imports.%Main.a_ptr_ref [concrete = imports.%a_ptr_ref.var]
// CHECK:STDOUT:   %.loc7: %ptr.c45 = acquire_value %a_ptr_ref.ref
// CHECK:STDOUT:   %impl.elem0.loc7: %.090 = impl_witness_access constants.%Copy.impl_witness.b9c, element0 [concrete = constants.%ptr.as.Copy.impl.Op.09c]
// CHECK:STDOUT:   %bound_method.loc7_23.1: <bound method> = bound_method %.loc7, %impl.elem0.loc7
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @ptr.as.Copy.impl.Op(constants.%const.0e5) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_23.2: <bound method> = bound_method %.loc7, %specific_fn.loc7
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc7: init %ptr.c45 = call %bound_method.loc7_23.2(%.loc7)
// CHECK:STDOUT:   assign file.%a_ptr.var, %ptr.as.Copy.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
