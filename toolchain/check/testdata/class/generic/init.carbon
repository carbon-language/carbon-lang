// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/init.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/init.carbon

// --- from_struct.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  var k: T;
}

//@dump-sem-ir-begin
fn InitFromStructGeneric(T:! Core.Copy, x: T) -> T {
  var v: Class(T) = {.k = x};
  return v.k;
}

fn InitFromStructSpecific(x: i32) -> i32 {
  var v: Class(i32) = {.k = x};
  return v.k;
}
//@dump-sem-ir-end

// --- adapt.carbon

library "[[@TEST_NAME]]";

class Adapt(T:! type) {
  adapt T;
}

//@dump-sem-ir-begin
fn InitFromAdaptedGeneric(T:! Core.Copy, x: T) -> T {
  return (x as Adapt(T)) as T;
}

fn InitFromAdaptedSpecific(x: i32) -> i32 {
  return (x as Adapt(i32)) as i32;
}
//@dump-sem-ir-end

// CHECK:STDOUT: --- from_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Class.b1d: type = class_type @Class, @Class(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Class.elem.73a: type = unbound_element_type %Class.b1d, %T.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k.c03: type = struct_type {.k: %T.as_type} [symbolic]
// CHECK:STDOUT:   %require_complete.57f: <witness> = require_complete_type %Class.b1d [symbolic]
// CHECK:STDOUT:   %pattern_type.a8f: type = pattern_type %Class.b1d [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %Destroy.impl_witness.492: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type.637: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.as_type) [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.598: %Class.as.Destroy.impl.Op.type.637 = struct_value () [symbolic]
// CHECK:STDOUT:   %Destroy.facet.744: %Destroy.type = facet_value %Class.b1d, (%Destroy.impl_witness.492) [symbolic]
// CHECK:STDOUT:   %.c08: type = fn_type_with_self_type %Destroy.Op.type, %Destroy.facet.744 [symbolic]
// CHECK:STDOUT:   %ptr.458: type = ptr_type %Class.b1d [symbolic]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn.df2: <specific function> = specific_function %Class.as.Destroy.impl.Op.598, @Class.as.Destroy.impl.Op(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.ec5: <witness> = require_complete_type %ptr.458 [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.247: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.2d8: type = unbound_element_type %Class.247, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.0fa: type = pattern_type %Class.247 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type.ce6: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%i32) [concrete]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.831: %Class.as.Destroy.impl.Op.type.ce6 = struct_value () [concrete]
// CHECK:STDOUT:   %ptr.f7c: type = ptr_type %Class.247 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.f8cebc.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.f8cebc.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_50: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc9_50: type = facet_access_type %T.ref.loc9_50 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_50: type = converted %T.ref.loc9_50, %T.as_type.loc9_50 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_34: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_26.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_26.1 (constants.%T.578)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_44.1: type = splice_block %.loc9_44.2 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_44: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc9_44.2: type = facet_access_type %T.ref.loc9_44 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_44.2: type = converted %T.ref.loc9_44, %T.as_type.loc9_44.2 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc14_38: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc14_38: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc14: type = splice_block %i32.loc14_30 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc14_30: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc14_30: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc9_26.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_26.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_26.1 (constants.%T.578)]
// CHECK:STDOUT:   %T.as_type.loc9_44.1: type = facet_access_type %T.loc9_26.1 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %T.as_type.loc9_44.1 [symbolic = %pattern_type.loc9 (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_44.1 [symbolic = %require_complete.loc9 (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%T.as_type.loc9_44.1) [symbolic = %Class.loc10_17.2 (constants.%Class.b1d)]
// CHECK:STDOUT:   %require_complete.loc10_17: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10_17 (constants.%require_complete.57f)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.a8f)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k.c03)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_26.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc9_44.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc10_27.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_27.2 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc10_27.2: @InitFromStructGeneric.%.loc10_27.2 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_27.2: <specific function> = specific_impl_function %impl.elem0.loc10_27.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc10_17.2, %T.as_type.loc9_44.1 [symbolic = %Class.elem (constants.%Class.elem.73a)]
// CHECK:STDOUT:   %Destroy.impl_witness: <witness> = impl_witness @Class.%Destroy.impl_witness_table, @Class.as.Destroy.impl(%T.as_type.loc9_44.1) [symbolic = %Destroy.impl_witness (constants.%Destroy.impl_witness.492)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc10_17.2, (%Destroy.impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.744)]
// CHECK:STDOUT:   %.loc10_3.2: type = fn_type_with_self_type constants.%Destroy.Op.type, %Destroy.facet [symbolic = %.loc10_3.2 (constants.%.c08)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.type: type = fn_type @Class.as.Destroy.impl.Op, @Class.as.Destroy.impl(%T.as_type.loc9_44.1) [symbolic = %Class.as.Destroy.impl.Op.type (constants.%Class.as.Destroy.impl.Op.type.637)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op: @InitFromStructGeneric.%Class.as.Destroy.impl.Op.type (%Class.as.Destroy.impl.Op.type.637) = struct_value () [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.598)]
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %Class.as.Destroy.impl.Op, @Class.as.Destroy.impl.Op(%T.as_type.loc9_44.1) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn.df2)]
// CHECK:STDOUT:   %ptr: type = ptr_type %Class.loc10_17.2 [symbolic = %ptr (constants.%ptr.458)]
// CHECK:STDOUT:   %require_complete.loc10_3: <witness> = require_complete_type %ptr [symbolic = %require_complete.loc10_3 (constants.%require_complete.ec5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type)) -> @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.a8f) = binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.a8f) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.b1d) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.c03) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc10_27.1: @InitFromStructGeneric.%.loc10_27.2 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc10_27.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_27.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_27.1: <specific function> = specific_impl_function %impl.elem0.loc10_27.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc10_27.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_27.1
// CHECK:STDOUT:     %.loc10_27.1: init @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = call %bound_method.loc10_27.2(%x.ref)
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = class_element_access %v.var, element0
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = initialize_from %.loc10_27.1 to %.loc10_28.2
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.b1d) = class_init (%.loc10_28.3), %v.var
// CHECK:STDOUT:     %.loc10_3.1: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.b1d) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.b1d)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: %Copy.type = name_ref T, %T.loc9_26.2 [symbolic = %T.loc9_26.1 (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc10_17.2: type = converted %T.ref.loc10, %T.as_type.loc10 [symbolic = %T.as_type.loc9_44.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%T.as_type) [symbolic = %Class.loc10_17.2 (constants.%Class.b1d)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.b1d) = bind_name v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.b1d) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.73a) = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc11_11.1: ref @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc11_11.2: @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = bind_value %.loc11_11.1
// CHECK:STDOUT:     %impl.elem0.loc11: @InitFromStructGeneric.%.loc10_27.2 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc11_11.1: <bound method> = bound_method %.loc11_11.2, %impl.elem0.loc11
// CHECK:STDOUT:     %specific_impl_fn.loc11: <specific function> = specific_impl_function %impl.elem0.loc11, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc11_11.2: <bound method> = bound_method %.loc11_11.2, %specific_impl_fn.loc11
// CHECK:STDOUT:     %.loc11_11.3: init @InitFromStructGeneric.%T.as_type.loc9_44.1 (%T.as_type) = call %bound_method.loc11_11.2(%.loc11_11.2)
// CHECK:STDOUT:     %impl.elem0.loc10_3: @InitFromStructGeneric.%.loc10_3.2 (%.c08) = impl_witness_access constants.%Destroy.impl_witness.492, element0 [symbolic = %Class.as.Destroy.impl.Op (constants.%Class.as.Destroy.impl.Op.598)]
// CHECK:STDOUT:     %bound_method.loc10_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc10_3
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0.loc10_3, @Class.as.Destroy.impl.Op(constants.%T.as_type) [symbolic = %Class.as.Destroy.impl.Op.specific_fn (constants.%Class.as.Destroy.impl.Op.specific_fn.df2)]
// CHECK:STDOUT:     %bound_method.loc10_3.2: <bound method> = bound_method %v.var, %specific_fn
// CHECK:STDOUT:     %addr: @InitFromStructGeneric.%ptr (%ptr.458) = addr_of %v.var
// CHECK:STDOUT:     %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc10_3.2(%addr)
// CHECK:STDOUT:     return %.loc11_11.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.0fa = binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.0fa = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.247 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_30.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc15: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %x.ref, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %x.ref, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15: init %i32 = call %bound_method.loc15_29.2(%x.ref)
// CHECK:STDOUT:   %.loc15_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc15_30.3: init %i32 = initialize_from %Int.as.Copy.impl.Op.call.loc15 to %.loc15_30.2
// CHECK:STDOUT:   %.loc15_30.4: init %Class.247 = class_init (%.loc15_30.3), %v.var
// CHECK:STDOUT:   %.loc15_3: init %Class.247 = converted %.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   assign %v.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_19: type = splice_block %Class [concrete = constants.%Class.247] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %int_32.loc15: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc15: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.247]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.247 = bind_name v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.247 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.2d8 = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = bind_value %.loc16_11.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc16_11.1: <bound method> = bound_method %.loc16_11.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_11.2: <bound method> = bound_method %.loc16_11.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_11.2(%.loc16_11.2)
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.bound: <bound method> = bound_method %v.var, constants.%Class.as.Destroy.impl.Op.831
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %bound_method.loc15_3: <bound method> = bound_method %v.var, %Class.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %addr: %ptr.f7c = addr_of %v.var
// CHECK:STDOUT:   %Class.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc15_3(%addr)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call.loc16 to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.578) {
// CHECK:STDOUT:   %T.loc9_26.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc9_44.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type.loc9 => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.9d1: type = class_type @Adapt, @Adapt(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.17e: <witness> = require_complete_type %Adapt.9d1 [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.526: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.857: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.6aa: %Int.as.Copy.impl.Op.type.857 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f0b: <witness> = impl_witness imports.%Copy.impl_witness_table.f59, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.af5: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.87e: %Int.as.Copy.impl.Op.type.af5 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.26d: %Copy.type = facet_value %i32, (%Copy.impl_witness.f0b) [concrete]
// CHECK:STDOUT:   %.3c4: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.26d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.87e, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.b3c: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.857) = import_ref Core//prelude/parts/int, loc17_31, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.6aa)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f59 = impl_witness_table (%Core.import_ref.b3c), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.f8cebc.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.f8cebc.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_51: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc9_51: type = facet_access_type %T.ref.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_51: type = converted %T.ref.loc9_51, %T.as_type.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_35: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_27.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_45.1: type = splice_block %.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_45: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc9_45.2: type = facet_access_type %T.ref.loc9_45 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_45.2: type = converted %T.ref.loc9_45, %T.as_type.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc13_39: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc13_39: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %.loc13: type = splice_block %i32.loc13_31 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc13_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc13_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %i32 = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc9_27.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_27.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:   %T.as_type.loc9_45.1: type = facet_access_type %T.loc9_27.1 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_45.1 [symbolic = %pattern_type (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_45.1 [symbolic = %require_complete.loc9 (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.as_type.loc9_45.1) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.9d1)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.17e)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc9_45.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc10_26.4: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc10_26.4 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: @InitFromAdaptedGeneric.%.loc10_26.4 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_26.2: <specific function> = specific_impl_function %impl.elem0.loc10_26.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type)) -> @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc10_23: type = facet_access_type %T.ref.loc10_22 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_23: type = converted %T.ref.loc10_22, %T.as_type.loc10_23 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.9d1)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.9d1) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.9d1) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: %Copy.type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc10_29: type = facet_access_type %T.ref.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_29: type = converted %T.ref.loc10_29, %T.as_type.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     %impl.elem0.loc10_26.1: @InitFromAdaptedGeneric.%.loc10_26.4 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %.loc10_26.2, %impl.elem0.loc10_26.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_26.1: <specific function> = specific_impl_function %impl.elem0.loc10_26.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc10_26.2: <bound method> = bound_method %.loc10_26.2, %specific_impl_fn.loc10_26.1
// CHECK:STDOUT:     %.loc10_26.3: init @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = call %bound_method.loc10_26.2(%.loc10_26.2)
// CHECK:STDOUT:     return %.loc10_26.3 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %int_32.loc14_22: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_22: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.526]
// CHECK:STDOUT:   %.loc14_13.1: %Adapt.526 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc14_13.2: %Adapt.526 = converted %x.ref, %.loc14_13.1
// CHECK:STDOUT:   %int_32.loc14_31: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32.loc14_31: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_28.1: %i32 = as_compatible %.loc14_13.2
// CHECK:STDOUT:   %.loc14_28.2: %i32 = converted %.loc14_13.2, %.loc14_28.1
// CHECK:STDOUT:   %impl.elem0: %.3c4 = impl_witness_access constants.%Copy.impl_witness.f0b, element0 [concrete = constants.%Int.as.Copy.impl.Op.87e]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %.loc14_28.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %.loc14_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_28.2(%.loc14_28.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.578) {
// CHECK:STDOUT:   %T.loc9_27.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc9_45.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
