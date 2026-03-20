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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Class.316: type = class_type @Class, @Class(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %Class.elem.765: type = unbound_element_type %Class.316, %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %struct_type.k.436: type = struct_type {.k: %T.binding.as_type} [symbolic]
// CHECK:STDOUT:   %require_complete.ae7: <witness> = require_complete_type %Class.316 [symbolic]
// CHECK:STDOUT:   %pattern_type.c54: type = pattern_type %Class.316 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.023 = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdc: <specific function> = specific_impl_function %impl.elem0.594, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Class.316, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.88b: %Destroy.type = facet_value %Class.316, (%Destroy.lookup_impl_witness) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.5c9: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.88b) [symbolic]
// CHECK:STDOUT:   %.0dd: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.5c9, %Destroy.facet.88b [symbolic]
// CHECK:STDOUT:   %impl.elem0.799: %.0dd = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d17: <specific function> = specific_impl_function %impl.elem0.799, @Destroy.WithSelf.Op(%Destroy.facet.88b) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class.805: type = class_type @Class, @Class(%i32) [concrete]
// CHECK:STDOUT:   %Class.elem.927: type = unbound_element_type %Class.805, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.1c2: type = pattern_type %Class.805 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type: type = fn_type @Destroy.Op [concrete]
// CHECK:STDOUT:   %Destroy.Op: %Destroy.Op.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc28 (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc28 (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc28 (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc28 (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc32.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc32: %Copy.type = name_ref T, %T.loc21.2 [symbolic = %T.loc21.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc32: type = facet_access_type %T.ref.loc32 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc32.3: type = converted %T.ref.loc32, %T.as_type.loc32 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc32.4: Core.Form = init_form %.loc32.3 [symbolic = %.loc32.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc24: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc21.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc21.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc29.1: type = splice_block %.loc29.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc29: %Copy.type = name_ref T, %T.loc21.2 [symbolic = %T.loc21.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc29: type = facet_access_type %T.ref.loc29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc29.2: type = converted %T.ref.loc29, %T.as_type.loc29 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc63 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc63: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc63: Core.Form = init_form %i32.loc63 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc60: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc21.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc21.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc21.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc21.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type.loc28: type = pattern_type %T.binding.as_type [symbolic = %pattern_type.loc28 (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc32.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc32.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc28: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc28 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Class.loc40.2: type = class_type @Class, @Class(%T.binding.as_type) [symbolic = %Class.loc40.2 (constants.%Class.316)]
// CHECK:STDOUT:   %require_complete.loc40: <witness> = require_complete_type %Class.loc40.2 [symbolic = %require_complete.loc40 (constants.%require_complete.ae7)]
// CHECK:STDOUT:   %pattern_type.loc35: type = pattern_type %Class.loc40.2 [symbolic = %pattern_type.loc35 (constants.%pattern_type.c54)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type)} [symbolic = %struct_type.k (constants.%struct_type.k.436)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc21.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc21.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc46: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc21.1 [symbolic = %.loc46 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc46.2: @InitFromStructGeneric.%.loc46 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc46.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc46.2: <specific function> = specific_impl_function %impl.elem0.loc46.2, @Copy.WithSelf.Op(%T.loc21.1) [symbolic = %specific_impl_fn.loc46.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc40.2, %T.binding.as_type [symbolic = %Class.elem (constants.%Class.elem.765)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness: <witness> = lookup_impl_witness %Class.loc40.2, @Destroy [symbolic = %Destroy.lookup_impl_witness (constants.%Destroy.lookup_impl_witness)]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %Class.loc40.2, (%Destroy.lookup_impl_witness) [symbolic = %Destroy.facet (constants.%Destroy.facet.88b)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.5c9)]
// CHECK:STDOUT:   %.loc34.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet [symbolic = %.loc34.2 (constants.%.0dd)]
// CHECK:STDOUT:   %impl.elem0.loc34.2: @InitFromStructGeneric.%.loc34.2 (%.0dd) = impl_witness_access %Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34.2 (constants.%impl.elem0.799)]
// CHECK:STDOUT:   %specific_impl_fn.loc34.2: <specific function> = specific_impl_function %impl.elem0.loc34.2, @Destroy.WithSelf.Op(%Destroy.facet) [symbolic = %specific_impl_fn.loc34.2 (constants.%specific_impl_fn.d17)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc35 (%pattern_type.c54) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc35 (%pattern_type.c54) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc40.2 (%Class.316) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc47.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.436) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc46.1: @InitFromStructGeneric.%.loc46 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc46.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc46.1: <bound method> = bound_method %x.ref, %impl.elem0.loc46.1
// CHECK:STDOUT:     %specific_impl_fn.loc46.1: <specific function> = specific_impl_function %impl.elem0.loc46.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc46.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc46.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc46.1
// CHECK:STDOUT:     %.loc47.2: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = class_element_access %v.var, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc46: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc47.2 = call %bound_method.loc46.2(%x.ref)
// CHECK:STDOUT:     %.loc47.3: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc47.2 = in_place_init %Copy.WithSelf.Op.call.loc46
// CHECK:STDOUT:     %.loc47.4: init @InitFromStructGeneric.%Class.loc40.2 (%Class.316) to %v.var = class_init (%.loc47.3)
// CHECK:STDOUT:     %.loc34.1: init @InitFromStructGeneric.%Class.loc40.2 (%Class.316) = converted %.loc47.1, %.loc47.4
// CHECK:STDOUT:     assign %v.var, %.loc34.1
// CHECK:STDOUT:     %.loc40.1: type = splice_block %Class.loc40.1 [symbolic = %Class.loc40.2 (constants.%Class.316)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc39: %Copy.type = name_ref T, %T.loc21.2 [symbolic = %T.loc21.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc40: type = facet_access_type %T.ref.loc39 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc40.2: type = converted %T.ref.loc39, %T.as_type.loc40 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %Class.loc40.1: type = class_type @Class, @Class(constants.%T.binding.as_type) [symbolic = %Class.loc40.2 (constants.%Class.316)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc40.2 (%Class.316) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc40.2 (%Class.316) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.765) = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:     %.loc51.1: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc51.2: @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = acquire_value %.loc51.1
// CHECK:STDOUT:     %impl.elem0.loc51: @InitFromStructGeneric.%.loc46 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc46.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc51.1: <bound method> = bound_method %.loc51.2, %impl.elem0.loc51
// CHECK:STDOUT:     %specific_impl_fn.loc51: <specific function> = specific_impl_function %impl.elem0.loc51, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc46.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc51.2: <bound method> = bound_method %.loc51.2, %specific_impl_fn.loc51
// CHECK:STDOUT:     %.loc32.1: ref @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc51: init @InitFromStructGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc32.1 = call %bound_method.loc51.2(%.loc51.2)
// CHECK:STDOUT:     %impl.elem0.loc34.1: @InitFromStructGeneric.%.loc34.2 (%.0dd) = impl_witness_access constants.%Destroy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc34.2 (constants.%impl.elem0.799)]
// CHECK:STDOUT:     %bound_method.loc34.1: <bound method> = bound_method %v.var, %impl.elem0.loc34.1
// CHECK:STDOUT:     %specific_impl_fn.loc34.1: <specific function> = specific_impl_function %impl.elem0.loc34.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.88b) [symbolic = %specific_impl_fn.loc34.2 (constants.%specific_impl_fn.d17)]
// CHECK:STDOUT:     %bound_method.loc34.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc34.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc34.2(%v.var)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc51 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.1c2 = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.1c2 = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.805 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc78.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc77: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc77.1: <bound method> = bound_method %x.ref, %impl.elem0.loc77
// CHECK:STDOUT:   %specific_fn.loc77: <specific function> = specific_function %impl.elem0.loc77, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc77.2: <bound method> = bound_method %x.ref, %specific_fn.loc77
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc77: init %i32 = call %bound_method.loc77.2(%x.ref)
// CHECK:STDOUT:   %.loc78.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc78.3: init %i32 to %.loc78.2 = in_place_init %Int.as.Copy.impl.Op.call.loc77
// CHECK:STDOUT:   %.loc78.4: init %Class.805 to %v.var = class_init (%.loc78.3)
// CHECK:STDOUT:   %.loc65: init %Class.805 = converted %.loc78.1, %.loc78.4
// CHECK:STDOUT:   assign %v.var, %.loc65
// CHECK:STDOUT:   %.loc71: type = splice_block %Class [concrete = constants.%Class.805] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32.loc70: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%i32) [concrete = constants.%Class.805]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.805 = ref_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.805 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.927 = name_ref k, @Class.%.loc14 [concrete = @Class.%.loc14]
// CHECK:STDOUT:   %.loc82.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc82.2: %i32 = acquire_value %.loc82.1
// CHECK:STDOUT:   %impl.elem0.loc82: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc82.1: <bound method> = bound_method %.loc82.2, %impl.elem0.loc82
// CHECK:STDOUT:   %specific_fn.loc82: <specific function> = specific_function %impl.elem0.loc82, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc82.2: <bound method> = bound_method %.loc82.2, %specific_fn.loc82
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc82: init %i32 = call %bound_method.loc82.2(%.loc82.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call.loc82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op(%self.param: ref %Class.805) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc21.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type.loc28 => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc32.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.f64: type = class_type @Adapt, @Adapt(%T.binding.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.888: <witness> = require_complete_type %Adapt.f64 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.023 = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdc: <specific function> = specific_impl_function %impl.elem0.594, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.808: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.824: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.9b9: %Int.as.Copy.impl.Op.type.824 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.f17: <witness> = impl_witness imports.%Copy.impl_witness_table.e76, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.546: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.664: %Int.as.Copy.impl.Op.type.546 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.f17) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.081: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.8e2: type = fn_type_with_self_type %Copy.WithSelf.Op.type.081, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.664, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.18d: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.824) = import_ref Core//prelude/parts/int, loc80, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.9b9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.e76 = impl_witness_table (%Core.import_ref.18d), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc30.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc30: %Copy.type = name_ref T, %T.loc19.2 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc30: type = facet_access_type %T.ref.loc30 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc30.3: type = converted %T.ref.loc30, %T.as_type.loc30 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc30.4: Core.Form = init_form %.loc30.3 [symbolic = %.loc30.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc22: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc19.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc27.1: type = splice_block %.loc27.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc27: %Copy.type = name_ref T, %T.loc19.2 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc27: type = facet_access_type %T.ref.loc27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc27.2: type = converted %T.ref.loc27, %T.as_type.loc27 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc53 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc53: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc53: Core.Form = init_form %i32.loc53 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc50: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc19.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc19.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc19.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc30.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc30.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc26: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete.loc26 (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Adapt.loc39.2: type = class_type @Adapt, @Adapt(%T.binding.as_type) [symbolic = %Adapt.loc39.2 (constants.%Adapt.f64)]
// CHECK:STDOUT:   %require_complete.loc35: <witness> = require_complete_type %Adapt.loc39.2 [symbolic = %require_complete.loc35 (constants.%require_complete.888)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc19.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc19.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc41.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc19.1 [symbolic = %.loc41.3 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc41.2: @InitFromAdaptedGeneric.%.loc41.3 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc41.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc41.2: <specific function> = specific_impl_function %impl.elem0.loc41.2, @Copy.WithSelf.Op(%T.loc19.1) [symbolic = %specific_impl_fn.loc41.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc38: %Copy.type = name_ref T, %T.loc19.2 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc39: type = facet_access_type %T.ref.loc38 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc39: type = converted %T.ref.loc38, %T.as_type.loc39 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %Adapt.loc39.1: type = class_type @Adapt, @Adapt(constants.%T.binding.as_type) [symbolic = %Adapt.loc39.2 (constants.%Adapt.f64)]
// CHECK:STDOUT:     %.loc35.1: @InitFromAdaptedGeneric.%Adapt.loc39.2 (%Adapt.f64) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc35.2: @InitFromAdaptedGeneric.%Adapt.loc39.2 (%Adapt.f64) = converted %x.ref, %.loc35.1
// CHECK:STDOUT:     %T.ref.loc42: %Copy.type = name_ref T, %T.loc19.2 [symbolic = %T.loc19.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc42: type = facet_access_type %T.ref.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc42: type = converted %T.ref.loc42, %T.as_type.loc42 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc41.1: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = as_compatible %.loc35.2
// CHECK:STDOUT:     %.loc41.2: @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = converted %.loc35.2, %.loc41.1
// CHECK:STDOUT:     %impl.elem0.loc41.1: @InitFromAdaptedGeneric.%.loc41.3 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc41.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc41.1: <bound method> = bound_method %.loc41.2, %impl.elem0.loc41.1
// CHECK:STDOUT:     %specific_impl_fn.loc41.1: <specific function> = specific_impl_function %impl.elem0.loc41.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc41.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc41.2: <bound method> = bound_method %.loc41.2, %specific_impl_fn.loc41.1
// CHECK:STDOUT:     %.loc30.1: ref @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @InitFromAdaptedGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc30.1 = call %bound_method.loc41.2(%.loc41.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %i32.loc61: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.808]
// CHECK:STDOUT:   %.loc58.1: %Adapt.808 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc58.2: %Adapt.808 = converted %x.ref, %.loc58.1
// CHECK:STDOUT:   %i32.loc65: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc64.1: %i32 = as_compatible %.loc58.2
// CHECK:STDOUT:   %.loc64.2: %i32 = converted %.loc58.2, %.loc64.1
// CHECK:STDOUT:   %impl.elem0: %.8e2 = impl_witness_access constants.%Copy.impl_witness.f17, element0 [concrete = constants.%Int.as.Copy.impl.Op.664]
// CHECK:STDOUT:   %bound_method.loc64.1: <bound method> = bound_method %.loc64.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc64.2: <bound method> = bound_method %.loc64.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc64.2(%.loc64.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc19.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc30.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
