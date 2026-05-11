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
// CHECK:STDOUT:   %BitAndWith.type.f2e: type = generic_interface_type @BitAndWith [concrete]
// CHECK:STDOUT:   %BitAndWith.generic: %BitAndWith.type.f2e = struct_value () [concrete]
// CHECK:STDOUT:   %BitAndWith.type.b10: type = facet_type <@BitAndWith, @BitAndWith(type)> [concrete]
// CHECK:STDOUT:   %BitAndWith.impl_witness: <witness> = impl_witness imports.%BitAndWith.impl_witness_table [concrete]
// CHECK:STDOUT:   %BitAndWith.facet: %BitAndWith.type.b10 = facet_value type, (%BitAndWith.impl_witness) [concrete]
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.type.584: type = fn_type @BitAndWith.WithSelf.Op, @BitAndWith.WithSelf(type, %BitAndWith.facet) [concrete]
// CHECK:STDOUT:   %.c0f: type = fn_type_with_self_type %BitAndWith.WithSelf.Op.type.584, %BitAndWith.facet [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.type: type = fn_type @type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op: %type.as.BitAndWith.impl.Op.type = struct_value () [concrete]
// CHECK:STDOUT:   %type.as.BitAndWith.impl.Op.bound: <bound method> = bound_method %Copy.type, %type.as.BitAndWith.impl.Op [concrete]
// CHECK:STDOUT:   %facet_type: type = facet_type <@Destroy & @Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.b13: type = pattern_type %facet_type [concrete]
// CHECK:STDOUT:   %T.65f: %facet_type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.db4: type = facet_access_type %T.65f [symbolic]
// CHECK:STDOUT:   %pattern_type.58d: type = pattern_type %T.as_type.db4 [symbolic]
// CHECK:STDOUT:   %.a57: Core.Form = init_form %T.as_type.db4 [symbolic]
// CHECK:STDOUT:   %InitFromStructGeneric.type: type = fn_type @InitFromStructGeneric [concrete]
// CHECK:STDOUT:   %InitFromStructGeneric: %InitFromStructGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.1e9: <witness> = require_complete_type %T.as_type.db4 [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.84a: <witness> = lookup_impl_witness %T.65f, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.5e1: %Destroy.type = facet_value %T.as_type.db4, (%Destroy.lookup_impl_witness.84a) [symbolic]
// CHECK:STDOUT:   %Class.46c: type = class_type @Class, @Class(%Destroy.facet.5e1) [symbolic]
// CHECK:STDOUT:   %Class.elem.ef9: type = unbound_element_type %Class.46c, %T.as_type.db4 [symbolic]
// CHECK:STDOUT:   %struct_type.k.92f: type = struct_type {.k: %T.as_type.db4} [symbolic]
// CHECK:STDOUT:   %require_complete.c1b: <witness> = require_complete_type %Class.46c [symbolic]
// CHECK:STDOUT:   %pattern_type.ebf: type = pattern_type %Class.46c [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.322: <witness> = lookup_impl_witness %T.65f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.856: %Copy.type = facet_value %T.as_type.db4, (%Copy.lookup_impl_witness.322) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.5d5: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.856) [symbolic]
// CHECK:STDOUT:   %.e56: type = fn_type_with_self_type %Copy.WithSelf.Op.type.5d5, %Copy.facet.856 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bab: %.e56 = impl_witness_access %Copy.lookup_impl_witness.322, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.421: <specific function> = specific_impl_function %impl.elem0.bab, @Copy.WithSelf.Op(%Copy.facet.856) [symbolic]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.488: <witness> = lookup_impl_witness %Class.46c, @Destroy [symbolic]
// CHECK:STDOUT:   %Destroy.facet.474: %Destroy.type = facet_value %Class.46c, (%Destroy.lookup_impl_witness.488) [symbolic]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.f65: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.474) [symbolic]
// CHECK:STDOUT:   %.1fe: type = fn_type_with_self_type %Destroy.WithSelf.Op.type.f65, %Destroy.facet.474 [symbolic]
// CHECK:STDOUT:   %impl.elem0.ae3: %.1fe = impl_witness_access %Destroy.lookup_impl_witness.488, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.4b3: <specific function> = specific_impl_function %impl.elem0.ae3, @Destroy.WithSelf.Op(%Destroy.facet.474) [symbolic]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %.ff5: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific.type: type = fn_type @InitFromStructSpecific [concrete]
// CHECK:STDOUT:   %InitFromStructSpecific: %InitFromStructSpecific.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32.builtin: type = int_type signed, %int_32 [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.2: type = fn_type @Destroy.Op.loc15_19.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.2: %Destroy.Op.type.bae255.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.8d7fae.2: <witness> = custom_witness (%Destroy.Op.651ba6.2), @Destroy [concrete]
// CHECK:STDOUT:   %Destroy.facet.0fd: %Destroy.type = facet_value %i32, (%custom_witness.8d7fae.2) [concrete]
// CHECK:STDOUT:   %Class.02d: type = class_type @Class, @Class(%Destroy.facet.0fd) [concrete]
// CHECK:STDOUT:   %Class.elem.2a7: type = unbound_element_type %Class.02d, %i32 [concrete]
// CHECK:STDOUT:   %struct_type.k.0bf: type = struct_type {.k: %i32} [concrete]
// CHECK:STDOUT:   %pattern_type.f2b: type = pattern_type %Class.02d [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.30d: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.aa9: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.44b: %Int.as.Copy.impl.Op.type.aa9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet.d1a: %Copy.type = facet_value %i32, (%Copy.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.cee: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.d1a) [concrete]
// CHECK:STDOUT:   %.348: type = fn_type_with_self_type %Copy.WithSelf.Op.type.cee, %Copy.facet.d1a [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.44b, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.bae255.4: type = fn_type @Destroy.Op.loc15_3.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.651ba6.4: %Destroy.Op.type.bae255.4 = struct_value () [concrete]
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
// CHECK:STDOUT:   %Core.BitAndWith: %BitAndWith.type.f2e = import_ref Core//prelude/parts/as, BitAndWith, loaded [concrete = constants.%BitAndWith.generic]
// CHECK:STDOUT:   %Core.import_ref.79e: %type.as.BitAndWith.impl.Op.type = import_ref Core//prelude/parts/as, loc{{\d+_\d+}}, loaded [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:   %BitAndWith.impl_witness_table = impl_witness_table (%Core.import_ref.79e), @type.as.BitAndWith.impl [concrete]
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromStructGeneric.decl: %InitFromStructGeneric.type = fn_decl @InitFromStructGeneric [concrete = constants.%InitFromStructGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.b13 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.58d) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.58d) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.58d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromStructGeneric.%pattern_type.loc9 (%pattern_type.58d) = return_slot_pattern %return.param_patt, %.loc9_65.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_65: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.65f)]
// CHECK:STDOUT:     %T.as_type.loc9_65: type = facet_access_type %T.ref.loc9_65 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:     %.loc9_65.3: type = converted %T.ref.loc9_65, %T.as_type.loc9_65 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:     %.loc9_65.4: Core.Form = init_form %.loc9_65.3 [symbolic = %.loc9_65.2 (constants.%.a57)]
// CHECK:STDOUT:     %.loc9_40.1: type = splice_block %.loc9_40.3 [concrete = constants.%facet_type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref.loc9_30: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:       %Core.ref.loc9_42: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Destroy.ref: type = name_ref Destroy, imports.%Core.Destroy [concrete = constants.%Destroy.type]
// CHECK:STDOUT:       %impl.elem0.loc9: %.c0f = impl_witness_access constants.%BitAndWith.impl_witness, element0 [concrete = constants.%type.as.BitAndWith.impl.Op]
// CHECK:STDOUT:       %bound_method.loc9: <bound method> = bound_method %Copy.ref, %impl.elem0.loc9 [concrete = constants.%type.as.BitAndWith.impl.Op.bound]
// CHECK:STDOUT:       %type.as.BitAndWith.impl.Op.call: init type = call %bound_method.loc9(%Copy.ref, %Destroy.ref) [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_40.2: type = value_of_initializer %type.as.BitAndWith.impl.Op.call [concrete = constants.%facet_type]
// CHECK:STDOUT:       %.loc9_40.3: type = converted %type.as.BitAndWith.impl.Op.call, %.loc9_40.2 [concrete = constants.%facet_type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_27.2: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.65f)]
// CHECK:STDOUT:     %x.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = value_param call_param0
// CHECK:STDOUT:     %.loc9_59.1: type = splice_block %.loc9_59.2 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)] {
// CHECK:STDOUT:       %T.ref.loc9_59: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.65f)]
// CHECK:STDOUT:       %T.as_type.loc9_59.2: type = facet_access_type %T.ref.loc9_59 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:       %.loc9_59.2: type = converted %T.ref.loc9_59, %T.as_type.loc9_59.2 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromStructSpecific.decl: %InitFromStructSpecific.type = fn_decl @InitFromStructSpecific [concrete = constants.%InitFromStructSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc14_38 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc14_38: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc14: Core.Form = init_form %i32.loc14_38 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc14_30: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromStructGeneric(%T.loc9_27.2: %facet_type) {
// CHECK:STDOUT:   %T.loc9_27.1: %facet_type = symbolic_binding T, 0 [symbolic = %T.loc9_27.1 (constants.%T.65f)]
// CHECK:STDOUT:   %T.as_type.loc9_59.1: type = facet_access_type %T.loc9_27.1 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:   %pattern_type.loc9: type = pattern_type %T.as_type.loc9_59.1 [symbolic = %pattern_type.loc9 (constants.%pattern_type.58d)]
// CHECK:STDOUT:   %.loc9_65.2: Core.Form = init_form %T.as_type.loc9_59.1 [symbolic = %.loc9_65.2 (constants.%.a57)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_59.1 [symbolic = %require_complete.loc9 (constants.%require_complete.1e9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_17: <witness> = lookup_impl_witness %T.loc9_27.1, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_17 (constants.%Destroy.lookup_impl_witness.84a)]
// CHECK:STDOUT:   %Destroy.facet.loc10_17.2: %Destroy.type = facet_value %T.as_type.loc9_59.1, (%Destroy.lookup_impl_witness.loc10_17) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.5e1)]
// CHECK:STDOUT:   %Class.loc10_17.2: type = class_type @Class, @Class(%Destroy.facet.loc10_17.2) [symbolic = %Class.loc10_17.2 (constants.%Class.46c)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Class.loc10_17.2 [symbolic = %require_complete.loc10 (constants.%require_complete.c1b)]
// CHECK:STDOUT:   %pattern_type.loc10: type = pattern_type %Class.loc10_17.2 [symbolic = %pattern_type.loc10 (constants.%pattern_type.ebf)]
// CHECK:STDOUT:   %struct_type.k: type = struct_type {.k: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4)} [symbolic = %struct_type.k (constants.%struct_type.k.92f)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_27.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.322)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc9_59.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.856)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.5d5)]
// CHECK:STDOUT:   %.loc10_27: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %Copy.facet [symbolic = %.loc10_27 (constants.%.e56)]
// CHECK:STDOUT:   %impl.elem0.loc10_27.2: @InitFromStructGeneric.%.loc10_27 (%.e56) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.bab)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_27.2: <specific function> = specific_impl_function %impl.elem0.loc10_27.2, @Copy.WithSelf.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.421)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class.loc10_17.2, %T.as_type.loc9_59.1 [symbolic = %Class.elem (constants.%Class.elem.ef9)]
// CHECK:STDOUT:   %Destroy.lookup_impl_witness.loc10_3: <witness> = lookup_impl_witness %Class.loc10_17.2, @Destroy [symbolic = %Destroy.lookup_impl_witness.loc10_3 (constants.%Destroy.lookup_impl_witness.488)]
// CHECK:STDOUT:   %Destroy.facet.loc10_3: %Destroy.type = facet_value %Class.loc10_17.2, (%Destroy.lookup_impl_witness.loc10_3) [symbolic = %Destroy.facet.loc10_3 (constants.%Destroy.facet.474)]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type: type = fn_type @Destroy.WithSelf.Op, @Destroy.WithSelf(%Destroy.facet.loc10_3) [symbolic = %Destroy.WithSelf.Op.type (constants.%Destroy.WithSelf.Op.type.f65)]
// CHECK:STDOUT:   %.loc10_3.2: type = fn_type_with_self_type %Destroy.WithSelf.Op.type, %Destroy.facet.loc10_3 [symbolic = %.loc10_3.2 (constants.%.1fe)]
// CHECK:STDOUT:   %impl.elem0.loc10_3.2: @InitFromStructGeneric.%.loc10_3.2 (%.1fe) = impl_witness_access %Destroy.lookup_impl_witness.loc10_3, element0 [symbolic = %impl.elem0.loc10_3.2 (constants.%impl.elem0.ae3)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_3.2: <specific function> = specific_impl_function %impl.elem0.loc10_3.2, @Destroy.WithSelf.Op(%Destroy.facet.loc10_3) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4)) -> out %return.param: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %v.patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.ebf) = ref_binding_pattern v [concrete]
// CHECK:STDOUT:       %v.var_patt: @InitFromStructGeneric.%pattern_type.loc10 (%pattern_type.ebf) = var_pattern %v.patt [concrete]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v.var: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.46c) = var %v.var_patt
// CHECK:STDOUT:     %x.ref: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = name_ref x, %x
// CHECK:STDOUT:     %.loc10_28.1: @InitFromStructGeneric.%struct_type.k (%struct_type.k.92f) = struct_literal (%x.ref)
// CHECK:STDOUT:     %impl.elem0.loc10_27.1: @InitFromStructGeneric.%.loc10_27 (%.e56) = impl_witness_access constants.%Copy.lookup_impl_witness.322, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.bab)]
// CHECK:STDOUT:     %bound_method.loc10_27.1: <bound method> = bound_method %x.ref, %impl.elem0.loc10_27.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_27.1: <specific function> = specific_impl_function %impl.elem0.loc10_27.1, @Copy.WithSelf.Op(constants.%Copy.facet.856) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.421)]
// CHECK:STDOUT:     %bound_method.loc10_27.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc10_27.1
// CHECK:STDOUT:     %.loc10_28.2: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = class_element_access %v.var, element0
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc10: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) to %.loc10_28.2 = call %bound_method.loc10_27.2(%x.ref)
// CHECK:STDOUT:     %.loc10_28.3: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) to %.loc10_28.2 = in_place_init %Copy.WithSelf.Op.call.loc10
// CHECK:STDOUT:     %.loc10_28.4: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.46c) to %v.var = class_init (%.loc10_28.3)
// CHECK:STDOUT:     %.loc10_3.1: init @InitFromStructGeneric.%Class.loc10_17.2 (%Class.46c) = converted %.loc10_28.1, %.loc10_28.4
// CHECK:STDOUT:     assign %v.var, %.loc10_3.1
// CHECK:STDOUT:     %.loc10_17.1: type = splice_block %Class.loc10_17.1 [symbolic = %Class.loc10_17.2 (constants.%Class.46c)] {
// CHECK:STDOUT:       %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:       %T.ref.loc10: %facet_type = name_ref T, %T.loc9_27.2 [symbolic = %T.loc9_27.1 (constants.%T.65f)]
// CHECK:STDOUT:       %T.as_type.loc10: type = facet_access_type %T.ref.loc10 [symbolic = %T.as_type.loc9_59.1 (constants.%T.as_type.db4)]
// CHECK:STDOUT:       %Destroy.facet.loc10_17.1: %Destroy.type = facet_value %T.as_type.loc10, (constants.%Destroy.lookup_impl_witness.84a) [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.5e1)]
// CHECK:STDOUT:       %.loc10_17.2: %Destroy.type = converted %T.ref.loc10, %Destroy.facet.loc10_17.1 [symbolic = %Destroy.facet.loc10_17.2 (constants.%Destroy.facet.5e1)]
// CHECK:STDOUT:       %Class.loc10_17.1: type = class_type @Class, @Class(constants.%Destroy.facet.5e1) [symbolic = %Class.loc10_17.2 (constants.%Class.46c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %v: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.46c) = ref_binding v, %v.var
// CHECK:STDOUT:     %v.ref: ref @InitFromStructGeneric.%Class.loc10_17.2 (%Class.46c) = name_ref v, %v
// CHECK:STDOUT:     %k.ref: @InitFromStructGeneric.%Class.elem (%Class.elem.ef9) = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:     %.loc11_11.1: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = class_element_access %v.ref, element0
// CHECK:STDOUT:     %.loc11_11.2: @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = acquire_value %.loc11_11.1
// CHECK:STDOUT:     %impl.elem0.loc11: @InitFromStructGeneric.%.loc10_27 (%.e56) = impl_witness_access constants.%Copy.lookup_impl_witness.322, element0 [symbolic = %impl.elem0.loc10_27.2 (constants.%impl.elem0.bab)]
// CHECK:STDOUT:     %bound_method.loc11_11.1: <bound method> = bound_method %.loc11_11.2, %impl.elem0.loc11
// CHECK:STDOUT:     %specific_impl_fn.loc11: <specific function> = specific_impl_function %impl.elem0.loc11, @Copy.WithSelf.Op(constants.%Copy.facet.856) [symbolic = %specific_impl_fn.loc10_27.2 (constants.%specific_impl_fn.421)]
// CHECK:STDOUT:     %bound_method.loc11_11.2: <bound method> = bound_method %.loc11_11.2, %specific_impl_fn.loc11
// CHECK:STDOUT:     %.loc9_65.1: ref @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call.loc11: init @InitFromStructGeneric.%T.as_type.loc9_59.1 (%T.as_type.db4) to %.loc9_65.1 = call %bound_method.loc11_11.2(%.loc11_11.2)
// CHECK:STDOUT:     %impl.elem0.loc10_3.1: @InitFromStructGeneric.%.loc10_3.2 (%.1fe) = impl_witness_access constants.%Destroy.lookup_impl_witness.488, element0 [symbolic = %impl.elem0.loc10_3.2 (constants.%impl.elem0.ae3)]
// CHECK:STDOUT:     %bound_method.loc10_3.1: <bound method> = bound_method %v.var, %impl.elem0.loc10_3.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_3.1: <specific function> = specific_impl_function %impl.elem0.loc10_3.1, @Destroy.WithSelf.Op(constants.%Destroy.facet.474) [symbolic = %specific_impl_fn.loc10_3.2 (constants.%specific_impl_fn.4b3)]
// CHECK:STDOUT:     %bound_method.loc10_3.2: <bound method> = bound_method %v.var, %specific_impl_fn.loc10_3.1
// CHECK:STDOUT:     %Destroy.WithSelf.Op.call: init %empty_tuple.type = call %bound_method.loc10_3.2(%v.var)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call.loc11 to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @InitFromStructSpecific(%x.param: %i32) -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %v.patt: %pattern_type.f2b = ref_binding_pattern v [concrete]
// CHECK:STDOUT:     %v.var_patt: %pattern_type.f2b = var_pattern %v.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v.var: ref %Class.02d = var %v.var_patt
// CHECK:STDOUT:   %x.ref: %i32 = name_ref x, %x
// CHECK:STDOUT:   %.loc15_30.1: %struct_type.k.0bf = struct_literal (%x.ref)
// CHECK:STDOUT:   %impl.elem0.loc15: %.348 = impl_witness_access constants.%Copy.impl_witness.30d, element0 [concrete = constants.%Int.as.Copy.impl.Op.44b]
// CHECK:STDOUT:   %bound_method.loc15_29.1: <bound method> = bound_method %x.ref, %impl.elem0.loc15
// CHECK:STDOUT:   %specific_fn.loc15: <specific function> = specific_function %impl.elem0.loc15, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc15_29.2: <bound method> = bound_method %x.ref, %specific_fn.loc15
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc15: init %i32 = call %bound_method.loc15_29.2(%x.ref)
// CHECK:STDOUT:   %.loc15_30.2: ref %i32 = class_element_access %v.var, element0
// CHECK:STDOUT:   %.loc15_30.3: init %i32 to %.loc15_30.2 = in_place_init %Int.as.Copy.impl.Op.call.loc15
// CHECK:STDOUT:   %.loc15_30.4: init %Class.02d to %v.var = class_init (%.loc15_30.3)
// CHECK:STDOUT:   %.loc15_3: init %Class.02d = converted %.loc15_30.1, %.loc15_30.4
// CHECK:STDOUT:   assign %v.var, %.loc15_3
// CHECK:STDOUT:   %.loc15_19.1: type = splice_block %Class [concrete = constants.%Class.02d] {
// CHECK:STDOUT:     %Class.ref: %Class.type = name_ref Class, file.%Class.decl [concrete = constants.%Class.generic]
// CHECK:STDOUT:     %i32.loc15: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %Destroy.facet: %Destroy.type = facet_value %i32.loc15, (constants.%custom_witness.8d7fae.2) [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:     %.loc15_19.2: %Destroy.type = converted %i32.loc15, %Destroy.facet [concrete = constants.%Destroy.facet.0fd]
// CHECK:STDOUT:     %Class: type = class_type @Class, @Class(constants.%Destroy.facet.0fd) [concrete = constants.%Class.02d]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %v: ref %Class.02d = ref_binding v, %v.var
// CHECK:STDOUT:   %v.ref: ref %Class.02d = name_ref v, %v
// CHECK:STDOUT:   %k.ref: %Class.elem.2a7 = name_ref k, @Class.%.loc5 [concrete = @Class.%.loc5]
// CHECK:STDOUT:   %.loc16_11.1: ref %i32 = class_element_access %v.ref, element0
// CHECK:STDOUT:   %.loc16_11.2: %i32 = acquire_value %.loc16_11.1
// CHECK:STDOUT:   %impl.elem0.loc16: %.348 = impl_witness_access constants.%Copy.impl_witness.30d, element0 [concrete = constants.%Int.as.Copy.impl.Op.44b]
// CHECK:STDOUT:   %bound_method.loc16_11.1: <bound method> = bound_method %.loc16_11.2, %impl.elem0.loc16
// CHECK:STDOUT:   %specific_fn.loc16: <specific function> = specific_function %impl.elem0.loc16, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc16_11.2: <bound method> = bound_method %.loc16_11.2, %specific_fn.loc16
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call.loc16: init %i32 = call %bound_method.loc16_11.2(%.loc16_11.2)
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %v.var, constants.%Destroy.Op.651ba6.4
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
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.1(%self.param: ref %struct_type.k.0bf) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc15_3.2(%self.param: ref %Class.02d) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromStructGeneric(constants.%T.65f) {
// CHECK:STDOUT:   %T.loc9_27.1 => constants.%T.65f
// CHECK:STDOUT:   %T.as_type.loc9_59.1 => constants.%T.as_type.db4
// CHECK:STDOUT:   %pattern_type.loc9 => constants.%pattern_type.58d
// CHECK:STDOUT:   %.loc9_65.2 => constants.%.a57
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
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.3b9bb5.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.435d17.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %InitFromAdaptedGeneric.type: type = fn_type @InitFromAdaptedGeneric [concrete]
// CHECK:STDOUT:   %InitFromAdaptedGeneric: %InitFromAdaptedGeneric.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.89e: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Adapt.2b1: type = class_type @Adapt, @Adapt(%T.as_type) [symbolic]
// CHECK:STDOUT:   %require_complete.619: <witness> = require_complete_type %Adapt.2b1 [symbolic]
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
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.161: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.012: %Int.as.Copy.impl.Op.type.161 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.30d: <witness> = impl_witness imports.%Copy.impl_witness_table.1ee, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.aa9: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.44b: %Int.as.Copy.impl.Op.type.aa9 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.30d) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.cee: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.348: type = fn_type_with_self_type %Copy.WithSelf.Op.type.cee, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.44b, @Int.as.Copy.impl.Op(%int_32) [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.dbb: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.161) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.012)]
// CHECK:STDOUT:   %Copy.impl_witness_table.1ee = impl_witness_table (%Core.import_ref.dbb), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %InitFromAdaptedGeneric.decl: %InitFromAdaptedGeneric.type = fn_decl @InitFromAdaptedGeneric [concrete = constants.%InitFromAdaptedGeneric] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.3b9bb5.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.3b9bb5.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.3b9bb5.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @InitFromAdaptedGeneric.%pattern_type (%pattern_type.3b9bb5.2) = return_slot_pattern %return.param_patt, %.loc9_51.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc9_51: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc9_51: type = facet_access_type %T.ref.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_51.3: type = converted %T.ref.loc9_51, %T.as_type.loc9_51 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc9_51.4: Core.Form = init_form %.loc9_51.3 [symbolic = %.loc9_51.2 (constants.%.435d17.2)]
// CHECK:STDOUT:     %.loc9_35: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc9_28.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc9_45.1: type = splice_block %.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc9_45: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc9_45.2: type = facet_access_type %T.ref.loc9_45 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc9_45.2: type = converted %T.ref.loc9_45, %T.as_type.loc9_45.2 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %InitFromAdaptedSpecific.decl: %InitFromAdaptedSpecific.type = fn_decl @InitFromAdaptedSpecific [concrete = constants.%InitFromAdaptedSpecific] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.7ce = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: %pattern_type.7ce = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern %return.param_patt, %i32.loc13_39 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc13_39: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc13: Core.Form = init_form %i32.loc13_39 [concrete = constants.%.ff5]
// CHECK:STDOUT:     %x.param: %i32 = value_param call_param0
// CHECK:STDOUT:     %i32.loc13_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %x: %i32 = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @InitFromAdaptedGeneric(%T.loc9_28.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc9_28.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.as_type.loc9_45.1: type = facet_access_type %T.loc9_28.1 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc9_45.1 [symbolic = %pattern_type (constants.%pattern_type.3b9bb5.2)]
// CHECK:STDOUT:   %.loc9_51.2: Core.Form = init_form %T.as_type.loc9_45.1 [symbolic = %.loc9_51.2 (constants.%.435d17.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc9: <witness> = require_complete_type %T.as_type.loc9_45.1 [symbolic = %require_complete.loc9 (constants.%require_complete.89e)]
// CHECK:STDOUT:   %Adapt.loc10_23.2: type = class_type @Adapt, @Adapt(%T.as_type.loc9_45.1) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.2b1)]
// CHECK:STDOUT:   %require_complete.loc10: <witness> = require_complete_type %Adapt.loc10_23.2 [symbolic = %require_complete.loc10 (constants.%require_complete.619)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc9_28.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc10_26.3: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc9_28.1 [symbolic = %.loc10_26.3 (constants.%.023)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc9_28.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %impl.elem0.loc10_26.2: @InitFromAdaptedGeneric.%.loc10_26.3 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc10_26.2: <specific function> = specific_impl_function %impl.elem0.loc10_26.2, @Copy.WithSelf.Op(%T.loc9_28.1) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type)) -> out %return.param: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %Adapt.ref: %Adapt.type = name_ref Adapt, file.%Adapt.decl [concrete = constants.%Adapt.generic]
// CHECK:STDOUT:     %T.ref.loc10_22: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc10_23: type = facet_access_type %T.ref.loc10_22 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_23: type = converted %T.ref.loc10_22, %T.as_type.loc10_23 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %Adapt.loc10_23.1: type = class_type @Adapt, @Adapt(constants.%T.as_type) [symbolic = %Adapt.loc10_23.2 (constants.%Adapt.2b1)]
// CHECK:STDOUT:     %.loc10_13.1: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.2b1) = as_compatible %x.ref
// CHECK:STDOUT:     %.loc10_13.2: @InitFromAdaptedGeneric.%Adapt.loc10_23.2 (%Adapt.2b1) = converted %x.ref, %.loc10_13.1
// CHECK:STDOUT:     %T.ref.loc10_29: %Copy.type = name_ref T, %T.loc9_28.2 [symbolic = %T.loc9_28.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc10_29: type = facet_access_type %T.ref.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_29: type = converted %T.ref.loc10_29, %T.as_type.loc10_29 [symbolic = %T.as_type.loc9_45.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc10_26.1: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = as_compatible %.loc10_13.2
// CHECK:STDOUT:     %.loc10_26.2: @InitFromAdaptedGeneric.%T.as_type.loc9_45.1 (%T.as_type) = converted %.loc10_13.2, %.loc10_26.1
// CHECK:STDOUT:     %impl.elem0.loc10_26.1: @InitFromAdaptedGeneric.%.loc10_26.3 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc10_26.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc10_26.1: <bound method> = bound_method %.loc10_26.2, %impl.elem0.loc10_26.1
// CHECK:STDOUT:     %specific_impl_fn.loc10_26.1: <specific function> = specific_impl_function %impl.elem0.loc10_26.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc10_26.2 (constants.%specific_impl_fn.bdc)]
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
// CHECK:STDOUT:   %Adapt: type = class_type @Adapt, @Adapt(constants.%i32) [concrete = constants.%Adapt.808]
// CHECK:STDOUT:   %.loc14_13.1: %Adapt.808 = as_compatible %x.ref
// CHECK:STDOUT:   %.loc14_13.2: %Adapt.808 = converted %x.ref, %.loc14_13.1
// CHECK:STDOUT:   %i32.loc14_31: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %.loc14_28.1: %i32 = as_compatible %.loc14_13.2
// CHECK:STDOUT:   %.loc14_28.2: %i32 = converted %.loc14_13.2, %.loc14_28.1
// CHECK:STDOUT:   %impl.elem0: %.348 = impl_witness_access constants.%Copy.impl_witness.30d, element0 [concrete = constants.%Int.as.Copy.impl.Op.44b]
// CHECK:STDOUT:   %bound_method.loc14_28.1: <bound method> = bound_method %.loc14_28.2, %impl.elem0
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc14_28.2: <bound method> = bound_method %.loc14_28.2, %specific_fn
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc14_28.2(%.loc14_28.2)
// CHECK:STDOUT:   return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @InitFromAdaptedGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc9_28.1 => constants.%T.035
// CHECK:STDOUT:   %T.as_type.loc9_45.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.3b9bb5.2
// CHECK:STDOUT:   %.loc9_51.2 => constants.%.435d17.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
