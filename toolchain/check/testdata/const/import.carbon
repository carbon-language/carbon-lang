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
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %T.d9f: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.897: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.d9f) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.990: %ptr.as.Copy.impl.Op.type.897 = struct_value () [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %const.668: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.801: type = ptr_type %const.668 [concrete]
// CHECK:STDOUT:   %pattern_type.c0d: type = pattern_type %ptr.801 [concrete]
// CHECK:STDOUT:   %pattern_type.6af: type = pattern_type %const.668 [concrete]
// CHECK:STDOUT:   %addr: %ptr.801 = addr_of imports.%a_ref.var [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.d5f: <witness> = impl_witness imports.%Copy.impl_witness_table.1a3, @ptr.as.Copy.impl(%const.668) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.22e: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%const.668) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.59b: %ptr.as.Copy.impl.Op.type.22e = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %ptr.801, (%Copy.impl_witness.d5f) [concrete]
// CHECK:STDOUT:   %.51a: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.bound: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.59b [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.59b, @ptr.as.Copy.impl.Op(%const.668) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %addr, %ptr.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.C: type = import_ref Main//implicit, C, loaded [concrete = constants.%C]
// CHECK:STDOUT:   %Main.a_ref: ref %const.668 = import_ref Main//implicit, a_ref, loaded [concrete = %a_ref.var]
// CHECK:STDOUT:   %Main.a_ptr_ref: ref %ptr.801 = import_ref Main//implicit, a_ptr_ref, loaded [concrete = %a_ptr_ref.var]
// CHECK:STDOUT:   %Main.import_ref.903: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.897) = import_ref Main//implicit, inst{{[0-9A-F]+}} [indirect], loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.990)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1a3 = impl_witness_table (%Main.import_ref.903), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %a_ref.patt: %pattern_type.6af = ref_binding_pattern a_ref [concrete]
// CHECK:STDOUT:   %a_ref.var_patt: %pattern_type.6af = var_pattern %a_ref.patt [concrete]
// CHECK:STDOUT:   %a_ref.var: ref %const.668 = var %a_ref.var_patt [concrete]
// CHECK:STDOUT:   %a_ptr_ref.patt: %pattern_type.c0d = ref_binding_pattern a_ptr_ref [concrete]
// CHECK:STDOUT:   %a_ptr_ref.var_patt: %pattern_type.c0d = var_pattern %a_ptr_ref.patt [concrete]
// CHECK:STDOUT:   %a_ptr_ref.var: ref %ptr.801 = var %a_ptr_ref.var_patt [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.c0d = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.c0d = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %ptr.801 = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc6: type = splice_block %ptr.loc6 [concrete = constants.%ptr.801] {
// CHECK:STDOUT:     %C.ref.loc6: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc6: type = const_type %C.ref.loc6 [concrete = constants.%const.668]
// CHECK:STDOUT:     %ptr.loc6: type = ptr_type %const.loc6 [concrete = constants.%ptr.801]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %ptr.801 = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_ptr.patt: %pattern_type.c0d = ref_binding_pattern a_ptr [concrete]
// CHECK:STDOUT:     %a_ptr.var_patt: %pattern_type.c0d = var_pattern %a_ptr.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr.var: ref %ptr.801 = var %a_ptr.var_patt [concrete]
// CHECK:STDOUT:   %.loc7: type = splice_block %ptr.loc7 [concrete = constants.%ptr.801] {
// CHECK:STDOUT:     %C.ref.loc7: type = name_ref C, imports.%Main.C [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc7: type = const_type %C.ref.loc7 [concrete = constants.%const.668]
// CHECK:STDOUT:     %ptr.loc7: type = ptr_type %const.loc7 [concrete = constants.%ptr.801]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_ptr: ref %ptr.801 = ref_binding a_ptr, %a_ptr.var [concrete = %a_ptr.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a_ref.ref: ref %const.668 = name_ref a_ref, imports.%Main.a_ref [concrete = imports.%a_ref.var]
// CHECK:STDOUT:   %addr: %ptr.801 = addr_of %a_ref.ref [concrete = constants.%addr]
// CHECK:STDOUT:   %impl.elem0.loc6: %.51a = impl_witness_access constants.%Copy.impl_witness.d5f, element0 [concrete = constants.%ptr.as.Copy.impl.Op.59b]
// CHECK:STDOUT:   %bound_method.loc6_19.1: <bound method> = bound_method %addr, %impl.elem0.loc6 [concrete = constants.%ptr.as.Copy.impl.Op.bound]
// CHECK:STDOUT:   %specific_fn.loc6: <specific function> = specific_function %impl.elem0.loc6, @ptr.as.Copy.impl.Op(constants.%const.668) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc6_19.2: <bound method> = bound_method %addr, %specific_fn.loc6 [concrete = constants.%bound_method]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc6: init %ptr.801 = call %bound_method.loc6_19.2(%addr) [concrete = constants.%addr]
// CHECK:STDOUT:   assign file.%a.var, %ptr.as.Copy.impl.Op.call.loc6
// CHECK:STDOUT:   %a_ptr_ref.ref: ref %ptr.801 = name_ref a_ptr_ref, imports.%Main.a_ptr_ref [concrete = imports.%a_ptr_ref.var]
// CHECK:STDOUT:   %.loc7: %ptr.801 = acquire_value %a_ptr_ref.ref
// CHECK:STDOUT:   %impl.elem0.loc7: %.51a = impl_witness_access constants.%Copy.impl_witness.d5f, element0 [concrete = constants.%ptr.as.Copy.impl.Op.59b]
// CHECK:STDOUT:   %bound_method.loc7_23.1: <bound method> = bound_method %.loc7, %impl.elem0.loc7
// CHECK:STDOUT:   %specific_fn.loc7: <specific function> = specific_function %impl.elem0.loc7, @ptr.as.Copy.impl.Op(constants.%const.668) [concrete = constants.%ptr.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc7_23.2: <bound method> = bound_method %.loc7, %specific_fn.loc7
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.call.loc7: init %ptr.801 = call %bound_method.loc7_23.2(%.loc7)
// CHECK:STDOUT:   assign file.%a_ptr.var, %ptr.as.Copy.impl.Op.call.loc7
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
