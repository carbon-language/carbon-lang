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

class Class(T:! Core.Destroy) {
  var k: T;
}

//@dump-sem-ir-begin
fn InitFromStructGeneric(T:! Core.Copy & Core.Destroy, x: T) -> T {
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
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %BitAndWith.type.a06: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.a06 = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.802: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.802 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.38f: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.d56: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.38f, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.709: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.283: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.74b: type = facet_access_type %T.283 [symbolic]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %.2b9: Core.Form = init_form %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.44d: <witness> = require_complete_type %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.28a: <witness> = lookup_impl_witness %T.283, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.16e: %Destroy.type = facet_value %T.as_type.74b, (%Destroy.lookup_impl_witness.28a) [symbolic]
// CHECK:STDOUT:   %Class.55b: type = class_type @Class, @Class(%Destroy.facet.16e) [symbolic]
// CHECK:STDOUT:   %Class.elem.b97: type = unbound_element_type %Class.55b, %T.as_type.74b [symbolic]
// CHECK:STDOUT:   %struct_type.k.aa4: type = struct_type {.k: %T.as_type.74b} [symbolic]
// CHECK:STDOUT:   %require_complete.ff5: <witness> = require_complete_type %Class.55b [symbolic]
// CHECK:STDOUT:   %pattern_type.0b6: type = pattern_type %Class.55b [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.4df: <witness> = lookup_impl_witness %T.283, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.3f0: %Copy.type = facet_value %T.as_type.74b, (%Copy.lookup_impl_witness.4df) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.090: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3f0) [symbolic]
// CHECK:STDOUT:   %.018: type = fn_type_with_self_type %Copy.WithSelf.Op.type.090, %Copy.facet.3f0 [symbolic]
// CHECK:STDOUT:   %impl.elem0.170: %.018 = impl_witness_access %Copy.lookup_impl_witness.4df, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.be5: <specific function> = specific_impl_function %impl.elem0.170, @Copy.WithSelf.Op(%Copy.facet.3f0) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.62e: <witness> = lookup_impl_witness %Class.55b, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.25e: %Destroy.type = facet_value %Class.55b, (%Destroy.lookup_impl_witness.62e) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.d76: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.25e) [symbolic]
// CHECK:STDOUT:   %.0da: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.d76, %Destroy.facet.25e [symbolic]
// CHECK:STDOUT:   %impl.elem0.730: %.0da = impl_witness_access %Destroy.lookup_impl_witness.62e, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fa9: <specific function> = specific_impl_function %impl.elem0.730, @Destroy.WithSelf.Op(%Destroy.facet.25e) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.2: type = fn_type @Destroy.Op.loc15_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.2: %Destroy.Op.type.1d8f74.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.df9cc1.2: <witness> = custom_witness (%Destroy.Op.1a2547.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.d07: %Destroy.type = facet_value %i32, (%custom_witness.df9cc1.2) [concrete]
// CHECK:STDOUT:   %Class.a32: type = class_type @Class, @Class(%Destroy.facet.d07) [concrete]
// CHECK:STDOUT:   %Class.elem.8f3: type = unbound_element_type %Class.a32, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.e7c: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.82c: type = pattern_type %Class.a32 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.987: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.987) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet.987 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.1d8f74.4: type = fn_type @Destroy.Op.loc15_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.1a2547.4: %Destroy.Op.type.1d8f74.4 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .BitAndWith = %Core.BitAndWith
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.a06 = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.9c3: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.9c3), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.709 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.44a) = return_slot_pattern %return.param_patt, %.loc9_65.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_65: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.283)]
// CHECK:STDOUT:     %T.as_type.loc9_65: type = facet_access_type %T.ref.loc9_65 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     %.loc9_65.3: type = converted %T.ref.loc9_65, %T.as_type.loc9_65 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     %.loc9_65.4: Core.Form = init_form %.loc9_65.3 [symbolic = %.loc9_65.2 (constants.%.2b9)]
// CHECK:STDOUT:     %.loc9_40.1: type = splice_block %.loc9_40.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc9_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc9_42: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc9: %.d56 = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc9: <bound method> = bound_method %Copy.ref, %impl.elem0.loc9 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc9(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_40.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_40.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc9_40.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_27.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.283)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = value_param call_param0
// CHECK:STDOUT:     %.loc9_59.1: type = splice_block %.loc9_59.2 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)] {
// CHECK:STDOUT:       %T.ref.loc9_59: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.283)]
// CHECK:STDOUT:       %T.as_type.loc9_59.2: type = facet_access_type %T.ref.loc9_59 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:       %.loc9_59.2: type = converted %T.ref.loc9_59, %T.as_type.loc9_59.2 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc14_38 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc14_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32.loc14_38 [concrete = constants.%.795]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc14_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc9_27.2: %facet_type) {
// CHECK:STDOUT:   %T.loc9_27.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.283)]
// CHECK:STDOUT:   %T.as_type.loc9_59.1: type = facet_access_type %T.loc9_27.1 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %T.as_type.loc9_59.1 [symbolic = %pattern_type.loc9 (constants.%pattern_type.44a)]
// CHECK:STDOUT:   %.loc9_65.2: Core.Form = init_form %T.as_type.loc9_59.1 [symbolic = %.loc9_65.2 (constants.%.2b9)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_59.1 [symbolic = %require_complete.loc9 (constants.%require_complete.44d)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_17: <witness> = lookup_impl_witness %T.loc9_27.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_17 (constants.%Destroy.lookup_impl_witness.28a)]
// CHECK:STDOUT:   %Destroy.facet.loc10_17.2: %Destroy.type = facet_value %T.as_type.loc9_59.1, (%Destroy.lookup_impl_witness.loc10_17) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%Destroy.facet.loc10_17.2) [symbolic = %Class.loc10_17.2 (constants.%Class.55b)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10 (constants.%require_complete.ff5)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.0b6)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b)} [symbolic = %struct_type.k (constants.%struct_type.k.aa4)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.4df)]
// CHECK:STDOUT:   %Copy.facet.loc10_27.2: %Copy.type = facet_value %T.as_type.loc9_59.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc10_27.2 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.loc10_27.2) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.090)]
// CHECK:STDOUT:   %.loc10_27.2: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet.loc10_27.2 [symbolic = %.loc10_27.2 (constants.%.018)]
// CHECK:STDOUT:   %impl.elem0.loc10_27.2: @InitFromStructGeneric.%.loc10_27.2 (%.018) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.170)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_27.2: <specific function> = specific_impl_function %impl.elem0.loc10_27.2, @Copy.WithSelf.Op(%Copy.facet.loc10_27.2) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.be5)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc10_17.2, %T.as_type.loc9_59.1 [symbolic = %Class.elem (constants.%Class.elem.b97)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_3: <witness> = lookup_impl_witness %Class.loc10_17.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_3 (constants.%Destroy.lookup_impl_witness.62e)]
// CHECK:STDOUT:   %Destroy.facet.loc10_3.2: %Destroy.type = facet_value %Class.loc10_17.2, (%Destroy.lookup_impl_witness.loc10_3) [symbolic = %Destroy.facet.loc10_3.2 (constants.%Destroy.facet.25e)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc10_3.2) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.d76)]
// CHECK:STDOUT:   %.loc10_3.3: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc10_3.2 [symbolic = %.loc10_3.3 (constants.%.0da)]
// CHECK:STDOUT:   %impl.elem0.loc10_3.2: @InitFromStructGeneric.%.loc10_3.3 (%.0da) = impl_witness_access %Destroy.lookup_impl_witness.loc10_3, element0 [symbolic = %impl.elem0.loc10_3.2 (constants.%impl.elem0.730)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_3.2: <specific function> = specific_impl_function %impl.elem0.loc10_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc10_3.2) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.fa9)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b)) -> out %return.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.0b6) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.0b6) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.55b) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.aa4) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc10_27.1: @InitFromStructGeneric.%.loc10_27.2 (%.018) = impl_witness_access constants.%Copy.lookup_impl_witness.4df, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.170)]
// CHECK:STDOUT:     %bound_method.loc10_27.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_27.1
// CHECK:STDOUT:     %Copy.facet.loc10_27.1: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.2 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc10_27.1: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc10_27.1 [symbolic = %Copy.facet.loc10_27.2 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_27.1: <specific function> = specific_impl_function %impl.elem0.loc10_27.1, @Copy.WithSelf.Op(constants.%Copy.facet.3f0) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.be5)]
// CHECK:STDOUT:     %bound_method.loc10_27.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_27.1
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = class_element_access %v.var, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc10: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) to %.loc10_28.2 = call %bound_method.loc10_27.2(%x.ref)
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) to %.loc10_28.2 = in_place_init %Copy.WithSelf.Op.call.loc10
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.55b) to %v.var = class_init (%.loc10_28.3)
// CHECK:STDOUT:     %.loc10_3.1: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.55b) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.55b)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.283)]
// CHECK:STDOUT:       %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.74b)]
// CHECK:STDOUT:       %Destroy.facet.loc10_17.1: %Destroy.type = facet_value %T.as_type.loc10, (constants.%Destroy.lookup_impl_witness.28a) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:       %.loc10_17.2: %Destroy.type = converted %T.ref.loc10, %Destroy.facet.loc10_17.1 [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.16e)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%Destroy.facet.16e) [symbolic = %Class.loc10_17.2 (constants.%Class.55b)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.55b) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.55b) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.b97) = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc11_11.1: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc11_11.2: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = acquire_value %.loc11_11.1
// CHECK:STDOUT:     %impl.elem0.loc11: @InitFromStructGeneric.%.loc10_27.2 (%.018) = impl_witness_access constants.%Copy.lookup_impl_witness.4df, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.170)]
// CHECK:STDOUT:     %bound_method.loc11_11.1: <bound method> = bound_method %.loc11_11.2, %impl.elem0.loc11
// CHECK:STDOUT:     %Copy.facet.loc11: %Copy.type = facet_value constants.%T.as_type.74b, (constants.%Copy.lookup_impl_witness.4df) [symbolic = %Copy.facet.loc10_27.2 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %.loc11_11.3: %Copy.type = converted constants.%T.as_type.74b, %Copy.facet.loc11 [symbolic = %Copy.facet.loc10_27.2 (constants.%Copy.facet.3f0)]
// CHECK:STDOUT:     %specific_impl_fn.loc11: <specific function> = specific_impl_function %impl.elem0.loc11, @Copy.WithSelf.Op(constants.%Copy.facet.3f0) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.be5)]
// CHECK:STDOUT:     %bound_method.loc11_11.2: <bound method> = bound_method %.loc11_11.2, %specific_impl_fn.loc11
// CHECK:STDOUT:     %.loc9_65.1: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc11: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.74b) to %.loc9_65.1 = call %bound_method.loc11_11.2(%.loc11_11.2)
// CHECK:STDOUT:     %impl.elem0.loc10_3.1: @InitFromStructGeneric.%.loc10_3.3 (%.0da) = impl_witness_access constants.%Destroy.lookup_impl_witness.62e, element0 [symbolic = %impl.elem0.loc10_3.2 (constants.%impl.elem0.730)]
// CHECK:STDOUT:     %bound_method.loc10_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc10_3.1
// CHECK:STDOUT:     %Destroy.facet.loc10_3.1: %Destroy.type = facet_value constants.%Class.55b, (constants.%Destroy.lookup_impl_witness.62e) [symbolic = %Destroy.facet.loc10_3.2 (constants.%Destroy.facet.25e)]
// CHECK:STDOUT:     %.loc10_3.2: %Destroy.type = converted constants.%Class.55b, %Destroy.facet.loc10_3.1 [symbolic = %Destroy.facet.loc10_3.2 (constants.%Destroy.facet.25e)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_3.1: <specific function> = specific_impl_function %impl.elem0.loc10_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.25e) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.fa9)]
// CHECK:STDOUT:     %bound_method.loc10_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc10_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc10_3.2(%v.var)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc11 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.82c = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.82c = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.a32 = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_30.1: %struct_type.k.e7c = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc15: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %x.ref, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %x.ref, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15: init %i32 = call %bound_method.loc15_29.2(%x.ref)
// CHECK:STDOUT:   %.loc15_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc15_30.3: init %i32 to %.loc15_30.2 = in_place_init %Int.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_30.4: init %Class.a32 to %v.var = class_init (%.loc15_30.3)
// CHECK:STDOUT:   %.loc15_3: init %Class.a32 = converted %.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   assign %v.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_19.1: type = splice_block %Class [concrete = constants.%Class.a32] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Destroy.facet: %Destroy.type = facet_value %i32.loc15, (constants.%custom_witness.df9cc1.2) [concrete = constants.%Destroy.facet.d07]
// CHECK:STDOUT:     %.loc15_19.2: %Destroy.type = converted %i32.loc15, %Destroy.facet [concrete = constants.%Destroy.facet.d07]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%Destroy.facet.d07) [concrete = constants.%Class.a32]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.a32 = ref_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.a32 = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.8f3 = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = acquire_value %.loc16_11.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc16_11.1: <bound method> = bound_method %.loc16_11.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_11.2: <bound method> = bound_method %.loc16_11.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_11.2(%.loc16_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.1a2547.4
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call %Destroy.Op.bound(%v.var)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call.loc16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_19.1(%self.param: ref %i32.builtin) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_19.2(%self.param: ref %i32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.1(%self.param: ref %struct_type.k.e7c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.2(%self.param: ref %Class.a32) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.283) {
// CHECK:STDOUT:   %T.loc9_27.1 => constants.%T.283
// CHECK:STDOUT:   %T.as_type.loc9_59.1 => constants.%T.as_type.74b
// CHECK:STDOUT:   %pattern_type.loc9 => constants.%pattern_type.44a
// CHECK:STDOUT:   %.loc9_65.2 => constants.%.2b9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- adapt.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Adapt.type: type = generic_class_type @Adapt [concrete]
// CHECK:STDOUT:   %Adapt.generic: %Adapt.type = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.210: type = class_type @Adapt, @Adapt(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.29a: <witness> = require_complete_type %Adapt.210 [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.6b6: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.795: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific.type: type = fn_type @InitFromAdaptedSpecific [concrete]
// CHECK:STDOUT:   %InitFromAdaptedSpecific: %InitFromAdaptedSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %Adapt.f89: type = class_type @Adapt, @Adapt(%i32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.ac8: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.5e0: %Int.as.Copy.impl.Op.type.ac8 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.0e0: <witness> = impl_witness imports.%Copy.impl_witness_table.367, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.3f6: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.4f6: %Int.as.Copy.impl.Op.type.3f6 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.0e0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d09: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.7a6: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d09, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.4f6, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.cd6: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.ac8) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.5e0)]
// CHECK:STDOUT:   %Copy.impl_witness_table.367 = impl_witness_table (%Core.import_ref.cd6), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e81 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt, %.loc9_51.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_51: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc9_51: type = facet_access_type %T.ref.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_51.3: type = converted %T.ref.loc9_51, %T.as_type.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_51.4: Core.Form = init_form %.loc9_51.3 [symbolic = %.loc9_51.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc9_35: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_28.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_45.1: type = splice_block %.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_45: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc9_45.2: type = facet_access_type %T.ref.loc9_45 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_45.2: type = converted %T.ref.loc9_45, %T.as_type.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.6b6 = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.6b6 = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.6b6 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.6b6 = return_slot_pattern %return.param_patt, %i32.loc13_39 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc13_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32.loc13_39 [concrete = constants.%.795]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc13_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc9_28.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_28.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc9_45.1: type = facet_access_type %T.loc9_28.1 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_45.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %.loc9_51.2: Core.Form = init_form %T.as_type.loc9_45.1 [symbolic = %.loc9_51.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_45.1 [symbolic = %require_complete.loc9 (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.as_type.loc9_45.1) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.210)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.29a)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc9_28.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc10_26.4: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc9_28.1 [symbolic = %.loc10_26.4 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: @InitFromAdaptedGeneric.%.loc10_26.4 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_26.2: <specific function> = specific_impl_function %impl.elem0.loc10_26.2, @Copy.WithSelf.Op(%T.loc9_28.1) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type)) -> out %return.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc10_23: type = facet_access_type %T.ref.loc10_22 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_23: type = converted %T.ref.loc10_22, %T.as_type.loc10_23 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.210)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.210) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.210) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc10_29: type = facet_access_type %T.ref.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_29: type = converted %T.ref.loc10_29, %T.as_type.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     %impl.elem0.loc10_26.1: @InitFromAdaptedGeneric.%.loc10_26.4 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %.loc10_26.2, %impl.elem0.loc10_26.1
// CHECK:STDOUT:     %.loc10_26.3: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc9_28.1 (constants.%T.f84)]
// CHECK:STDOUT:     %specific_impl_fn.loc10_26.1: <specific function> = specific_impl_function %impl.elem0.loc10_26.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc10_26.2: <bound method> = bound_method %.loc10_26.2, %specific_impl_fn.loc10_26.1
// CHECK:STDOUT:     %.loc9_51.1: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) to %.loc9_51.1 = call %bound_method.loc10_26.2(%.loc10_26.2)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromAdaptedSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:   %i32.loc14_22: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.f89]
// CHECK:STDOUT:   %.loc14_13.1: %Adapt.f89 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc14_13.2: %Adapt.f89 = converted %x.ref, %.loc14_13.1
// CHECK:STDOUT:   %i32.loc14_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_28.1: %i32 = as_compatible %.loc14_13.2
// CHECK:STDOUT:   %.loc14_28.2: %i32 = converted %.loc14_13.2, %.loc14_28.1
// CHECK:STDOUT:   %impl.elem0: %.7a6 = impl_witness_access constants.%Copy.impl_witness.0e0, element0 [concrete = constants.%Int.as.Copy.impl.Op.4f6]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %.loc14_28.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %.loc14_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_28.2(%.loc14_28.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc9_28.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc9_45.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %.loc9_51.2 => constants.%.1ce700.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
