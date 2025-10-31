// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/builtins/type/destroy.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/builtins/type/destroy.carbon

// --- forall.carbon
library "[[@TEST_NAME]]";

interface DestroyLike {
  fn Op[addr self: Self*]();
}

impl forall [T:! type] T as DestroyLike {
  fn Op[addr self: Self*]() = "type.destroy";
}

var a: ();
var b: {};

fn F() {
  //@dump-sem-ir-begin
  a.(DestroyLike.Op)();
  b.(DestroyLike.Op)();
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- forall.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %DestroyLike.type: type = facet_type <@DestroyLike> [concrete]
// CHECK:STDOUT:   %DestroyLike.Op.type: type = fn_type @DestroyLike.Op [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %DestroyLike.assoc_type: type = assoc_entity_type @DestroyLike [concrete]
// CHECK:STDOUT:   %assoc0: %DestroyLike.assoc_type = assoc_entity element0, @DestroyLike.%DestroyLike.Op.decl [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %DestroyLike.impl_witness.707: <witness> = impl_witness file.%DestroyLike.impl_witness_table, @T.as.DestroyLike.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.type.8d2: type = fn_type @T.as.DestroyLike.impl.Op, @T.as.DestroyLike.impl(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.caa: %T.as.DestroyLike.impl.Op.type.8d2 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.843: type = ptr_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %DestroyLike.facet.c03: %DestroyLike.type = facet_value %empty_tuple.type, (%DestroyLike.impl_witness.707) [concrete]
// CHECK:STDOUT:   %.f8b: type = fn_type_with_self_type %DestroyLike.Op.type, %DestroyLike.facet.c03 [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.bound.0a7: <bound method> = bound_method file.%a.var, %T.as.DestroyLike.impl.Op.caa [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.specific_fn.211: <specific function> = specific_function %T.as.DestroyLike.impl.Op.caa, @T.as.DestroyLike.impl.Op(%empty_tuple.type) [concrete]
// CHECK:STDOUT:   %bound_method.891: <bound method> = bound_method file.%a.var, %T.as.DestroyLike.impl.Op.specific_fn.211 [concrete]
// CHECK:STDOUT:   %addr.feb: %ptr.843 = addr_of file.%a.var [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %DestroyLike.impl_witness.97d: <witness> = impl_witness file.%DestroyLike.impl_witness_table, @T.as.DestroyLike.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.type.513: type = fn_type @T.as.DestroyLike.impl.Op, @T.as.DestroyLike.impl(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.e5b: %T.as.DestroyLike.impl.Op.type.513 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.c28: type = ptr_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %DestroyLike.facet.e73: %DestroyLike.type = facet_value %empty_struct_type, (%DestroyLike.impl_witness.97d) [concrete]
// CHECK:STDOUT:   %.642: type = fn_type_with_self_type %DestroyLike.Op.type, %DestroyLike.facet.e73 [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.bound.fba: <bound method> = bound_method file.%b.var, %T.as.DestroyLike.impl.Op.e5b [concrete]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.specific_fn.904: <specific function> = specific_function %T.as.DestroyLike.impl.Op.e5b, @T.as.DestroyLike.impl.Op(%empty_struct_type) [concrete]
// CHECK:STDOUT:   %bound_method.d4a: <bound method> = bound_method file.%b.var, %T.as.DestroyLike.impl.Op.specific_fn.904 [concrete]
// CHECK:STDOUT:   %addr.6aa: %ptr.c28 = addr_of file.%b.var [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: ref %empty_tuple.type = name_ref a, file.%a [concrete = file.%a.var]
// CHECK:STDOUT:   %DestroyLike.ref.loc16: type = name_ref DestroyLike, file.%DestroyLike.decl [concrete = constants.%DestroyLike.type]
// CHECK:STDOUT:   %Op.ref.loc16: %DestroyLike.assoc_type = name_ref Op, @DestroyLike.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc16: %.f8b = impl_witness_access constants.%DestroyLike.impl_witness.707, element0 [concrete = constants.%T.as.DestroyLike.impl.Op.caa]
// CHECK:STDOUT:   %bound_method.loc16_4: <bound method> = bound_method %a.ref, %impl.elem0.loc16 [concrete = constants.%T.as.DestroyLike.impl.Op.bound.0a7]
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @T.as.DestroyLike.impl.Op(constants.%empty_tuple.type) [concrete = constants.%T.as.DestroyLike.impl.Op.specific_fn.211]
// CHECK:STDOUT:   %bound_method.loc16_22: <bound method> = bound_method %a.ref, %specific_fn.loc16 [concrete = constants.%bound_method.891]
// CHECK:STDOUT:   %addr.loc16: %ptr.843 = addr_of %a.ref [concrete = constants.%addr.feb]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.call.loc16: init %empty_tuple.type = call %bound_method.loc16_22(%addr.loc16) [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %b.ref: ref %empty_struct_type = name_ref b, file.%b [concrete = file.%b.var]
// CHECK:STDOUT:   %DestroyLike.ref.loc17: type = name_ref DestroyLike, file.%DestroyLike.decl [concrete = constants.%DestroyLike.type]
// CHECK:STDOUT:   %Op.ref.loc17: %DestroyLike.assoc_type = name_ref Op, @DestroyLike.%assoc0 [concrete = constants.%assoc0]
// CHECK:STDOUT:   %impl.elem0.loc17: %.642 = impl_witness_access constants.%DestroyLike.impl_witness.97d, element0 [concrete = constants.%T.as.DestroyLike.impl.Op.e5b]
// CHECK:STDOUT:   %bound_method.loc17_4: <bound method> = bound_method %b.ref, %impl.elem0.loc17 [concrete = constants.%T.as.DestroyLike.impl.Op.bound.fba]
// CHECK:STDOUT:   %specific_fn.loc17: <specific function> = specific_function %impl.elem0.loc17, @T.as.DestroyLike.impl.Op(constants.%empty_struct_type) [concrete = constants.%T.as.DestroyLike.impl.Op.specific_fn.904]
// CHECK:STDOUT:   %bound_method.loc17_22: <bound method> = bound_method %b.ref, %specific_fn.loc17 [concrete = constants.%bound_method.d4a]
// CHECK:STDOUT:   %addr.loc17: %ptr.c28 = addr_of %b.ref [concrete = constants.%addr.6aa]
// CHECK:STDOUT:   %T.as.DestroyLike.impl.Op.call.loc17: init %empty_tuple.type = call %bound_method.loc17_22(%addr.loc17) [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
