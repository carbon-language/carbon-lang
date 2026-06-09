// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/generic/call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/generic/call.carbon

// --- explicit.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn Function(T:! Core.Copy, x: T) -> T {
  return x;
}
//@dump-sem-ir-end

fn CallGeneric(T:! Core.Copy, x: T) -> T {
  //@dump-sem-ir-begin
  return Function(T, x);
  //@dump-sem-ir-end
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  //@dump-sem-ir-begin
  return Function(T*, x);
  //@dump-sem-ir-end
}

class C {}

fn CallSpecific(x: C*) -> C* {
  //@dump-sem-ir-begin
  return Function(C*, x);
  //@dump-sem-ir-end
}

// --- deduced.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
fn Function[T:! Core.Copy](x: T) -> T {
  return x;
}
//@dump-sem-ir-end

fn CallGeneric(T:! Core.Copy, x: T) -> T {
  //@dump-sem-ir-begin
  return Function(x);
  //@dump-sem-ir-end
}

fn CallGenericPtr(T:! type, x: T*) -> T* {
  //@dump-sem-ir-begin
  return Function(x);
  //@dump-sem-ir-end
}

class C {}

fn CallSpecific(x: C*) -> C* {
  //@dump-sem-ir-begin
  return Function(x);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- explicit.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4db: <specific function> = specific_function %Function, @Function(%T.f84) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.162: <specific function> = specific_function %Function, @Function(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %.8fa: Core.Form = init_form %ptr.6b6 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.65f: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b22: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.c19: %ptr.as.Copy.impl.Op.type.b22 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.bf9: <witness> = complete_type_witness %ptr.6b6 [concrete]
// CHECK:STDOUT:   %Copy.facet.3f0: %Copy.type = facet_value %ptr.6b6, (%Copy.impl_witness.65f) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.710: <specific function> = specific_function %Function, @Function(%Copy.facet.3f0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c6a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3f0) [concrete]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c6a, %Copy.facet.3f0 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.c19, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e81 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt, %.loc5_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_37: %Copy.type = name_ref T, %T.loc5_14.2 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc5_37: type = facet_access_type %T.ref.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37.3: type = converted %T.ref.loc5_37, %T.as_type.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37.4: Core.Form = init_form %.loc5_37.3 [symbolic = %.loc5_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc5_21: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_31.1: type = splice_block %.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_31: %Copy.type = name_ref T, %T.loc5_14.2 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc5_31.2: type = facet_access_type %T.ref.loc5_31 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_31.2: type = converted %T.ref.loc5_31, %T.as_type.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc5_31.1: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_31.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %.loc5_37.2: Core.Form = init_form %T.as_type.loc5_31.1 [symbolic = %.loc5_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_31.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc5_14.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc5_14.1 [symbolic = %.loc6 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_14.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @Function.%.loc6 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.WithSelf.Op(%T.loc5_14.1) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type)) -> out %return.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.as_type.loc5_31.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @Function.%.loc6 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %x.ref, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc5_37.1: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Function.%T.as_type.loc5_31.1 (%T.as_type) to %.loc5_37.1 = call %bound_method.loc6_10.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc10_17.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc12_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc10_17.1) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.4db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type)) -> out %return.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc12: %Copy.type = name_ref T, %T.loc10_17.2 [symbolic = %T.loc10_17.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc12: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc10_17.1 (constants.%T.f84)]
// CHECK:STDOUT:     %Function.specific_fn.loc12_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.f84) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.4db)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) to %.loc10_40.1 = call %Function.specific_fn.loc12_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc16_20.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc18_24.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc16_20.1) [symbolic = %.loc18_24.3 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_33.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc18_24.3: %Copy.type = facet_value %ptr.loc16_33.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc18_24.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Function.specific_fn.loc18_10.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc18_24.3) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.162)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f)) -> out %return.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc18: type = name_ref T, %T.loc16_20.2 [symbolic = %T.loc16_20.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %T.ref.loc18 [symbolic = %ptr.loc16_33.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc18_24.1: %Copy.type = facet_value %ptr.loc18, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc18_24.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc18_24.1: %Copy.type = converted %ptr.loc18, %Copy.facet.loc18_24.1 [symbolic = %Copy.facet.loc18_24.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Copy.facet.loc18_24.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc18_24.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc18_24.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc18_24.2 [symbolic = %Copy.facet.loc18_24.3 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Function.specific_fn.loc18_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.302) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.162)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) = call %Function.specific_fn.loc18_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.6b6) -> out %return.param: %ptr.6b6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc26: type = ptr_type %C.ref.loc26 [concrete = constants.%ptr.6b6]
// CHECK:STDOUT:   %x.ref: %ptr.6b6 = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet.loc26_24.1: %Copy.type = facet_value %ptr.loc26, (constants.%Copy.impl_witness.65f) [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %.loc26_24.1: %Copy.type = converted %ptr.loc26, %Copy.facet.loc26_24.1 [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %Copy.facet.loc26_24.2: %Copy.type = facet_value constants.%ptr.6b6, (constants.%Copy.impl_witness.65f) [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %.loc26_24.2: %Copy.type = converted constants.%ptr.6b6, %Copy.facet.loc26_24.2 [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.3f0) [concrete = constants.%Function.specific_fn.710]
// CHECK:STDOUT:   %Function.call: init %ptr.6b6 = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.1ce700.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c85f21.2
// CHECK:STDOUT:   %.loc6 => constants.%.c29
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.d9c
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.bd2
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.e2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc10_17.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc10_34.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %.loc10_40.2 => constants.%.1ce700.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc16_20.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc16_33.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc16_40.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.302) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%Copy.facet.302
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.884
// CHECK:STDOUT:   %.loc6 => constants.%.be5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.1da
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.484
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.3f0) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%Copy.facet.3f0
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.6b6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcb
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.8fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.bf9
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c6a
// CHECK:STDOUT:   %.loc6 => constants.%.a68
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.65f
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.c19
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.e81: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.f84: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.f84 [symbolic]
// CHECK:STDOUT:   %pattern_type.2c66b4.2: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %.1ce700.2: Core.Form = init_form %T.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.d83: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.d9c: <witness> = lookup_impl_witness %T.f84, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c85f21.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.f84) [symbolic]
// CHECK:STDOUT:   %.c29: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c85f21.2, %T.f84 [symbolic]
// CHECK:STDOUT:   %impl.elem0.bd2: %.c29 = impl_witness_access %Copy.lookup_impl_witness.d9c, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.e2a: <specific function> = specific_impl_function %impl.elem0.bd2, @Copy.WithSelf.Op(%T.f84) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.ff5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ce9: %ptr.as.Copy.impl.Op.type.ff5 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.4db: <specific function> = specific_function %Function, @Function(%T.f84) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.1da: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %.0e9: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.facet.302: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.1da) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.162: <specific function> = specific_function %Function, @Function(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.6b6: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.fcb: type = pattern_type %ptr.6b6 [concrete]
// CHECK:STDOUT:   %.8fa: Core.Form = init_form %ptr.6b6 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.65f: <witness> = impl_witness imports.%Copy.impl_witness_table.4f1, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.b22: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.c19: %ptr.as.Copy.impl.Op.type.b22 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.bf9: <witness> = complete_type_witness %ptr.6b6 [concrete]
// CHECK:STDOUT:   %Copy.facet.3f0: %Copy.type = facet_value %ptr.6b6, (%Copy.impl_witness.65f) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.710: <specific function> = specific_function %Function, @Function(%Copy.facet.3f0) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.884: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %.be5: type = fn_type_with_self_type %Copy.WithSelf.Op.type.884, %Copy.facet.302 [symbolic]
// CHECK:STDOUT:   %impl.elem0.484: %.be5 = impl_witness_access %Copy.lookup_impl_witness.1da, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.a76: <specific function> = specific_impl_function %impl.elem0.484, @Copy.WithSelf.Op(%Copy.facet.302) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.c6a: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.3f0) [concrete]
// CHECK:STDOUT:   %.a68: type = fn_type_with_self_type %Copy.WithSelf.Op.type.c6a, %Copy.facet.3f0 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.c19, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.e5f: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.ff5) = import_ref Core//prelude/parts/copy, loc{{\d+_\d+}}, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.ce9)]
// CHECK:STDOUT:   %Copy.impl_witness_table.4f1 = impl_witness_table (%Core.import_ref.e5f), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.e81 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.2c66b4.2) = return_slot_pattern %return.param_patt, %.loc5_37.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_37: %Copy.type = name_ref T, %T.loc5_14.2 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:     %T.as_type.loc5_37: type = facet_access_type %T.ref.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37.3: type = converted %T.ref.loc5_37, %T.as_type.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37.4: Core.Form = init_form %.loc5_37.3 [symbolic = %.loc5_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:     %.loc5_21: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_14.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:     %x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_31.1: type = splice_block %.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_31: %Copy.type = name_ref T, %T.loc5_14.2 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:       %T.as_type.loc5_31.2: type = facet_access_type %T.ref.loc5_31 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_31.2: type = converted %T.ref.loc5_31, %T.as_type.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc5_14.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_14.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc5_14.1 (constants.%T.f84)]
// CHECK:STDOUT:   %T.as_type.loc5_31.1: type = facet_access_type %T.loc5_14.1 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_31.1 [symbolic = %pattern_type (constants.%pattern_type.2c66b4.2)]
// CHECK:STDOUT:   %.loc5_37.2: Core.Form = init_form %T.as_type.loc5_31.1 [symbolic = %.loc5_37.2 (constants.%.1ce700.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_31.1 [symbolic = %require_complete (constants.%require_complete.d83)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc5_14.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.c85f21.2)]
// CHECK:STDOUT:   %.loc6: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc5_14.1 [symbolic = %.loc6 (constants.%.c29)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_14.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.d9c)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @Function.%.loc6 (%.c29) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.WithSelf.Op(%T.loc5_14.1) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type)) -> out %return.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.as_type.loc5_31.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @Function.%.loc6 (%.c29) = impl_witness_access constants.%Copy.lookup_impl_witness.d9c, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.bd2)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %x.ref, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.WithSelf.Op(constants.%T.f84) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.e2a)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc5_37.1: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Function.%T.as_type.loc5_31.1 (%T.as_type) to %.loc5_37.1 = call %bound_method.loc6_10.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc10_17.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc12_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc10_17.1) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.4db)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type)) -> out %return.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc12: %Copy.type = converted constants.%T.as_type, constants.%T.f84 [symbolic = %T.loc10_17.1 (constants.%T.f84)]
// CHECK:STDOUT:     %Function.specific_fn.loc12_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.f84) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.4db)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) to %.loc10_40.1 = call %Function.specific_fn.loc12_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc16_20.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc18_20.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc16_20.1) [symbolic = %.loc18_20.2 (constants.%.0e9)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_33.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.1da)]
// CHECK:STDOUT:   %Copy.facet.loc18_20.2: %Copy.type = facet_value %ptr.loc16_33.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc18_20.2 (constants.%Copy.facet.302)]
// CHECK:STDOUT:   %Function.specific_fn.loc18_10.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc18_20.2) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.162)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f)) -> out %return.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc18_20.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.1da) [symbolic = %Copy.facet.loc18_20.2 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %.loc18_20.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc18_20.1 [symbolic = %Copy.facet.loc18_20.2 (constants.%Copy.facet.302)]
// CHECK:STDOUT:     %Function.specific_fn.loc18_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.302) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.162)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc16_33.1 (%ptr.e8f) = call %Function.specific_fn.loc18_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.6b6) -> out %return.param: %ptr.6b6 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %x.ref: %ptr.6b6 = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%ptr.6b6, (constants.%Copy.impl_witness.65f) [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %.loc26: %Copy.type = converted constants.%ptr.6b6, %Copy.facet [concrete = constants.%Copy.facet.3f0]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.3f0) [concrete = constants.%Function.specific_fn.710]
// CHECK:STDOUT:   %Function.call: init %ptr.6b6 = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.1ce700.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.d83
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c85f21.2
// CHECK:STDOUT:   %.loc6 => constants.%.c29
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.d9c
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.bd2
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.e2a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.f84) {
// CHECK:STDOUT:   %T.loc10_17.1 => constants.%T.f84
// CHECK:STDOUT:   %T.as_type.loc10_34.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.2c66b4.2
// CHECK:STDOUT:   %.loc10_40.2 => constants.%.1ce700.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc16_20.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc16_33.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc16_40.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.302) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%Copy.facet.302
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.884
// CHECK:STDOUT:   %.loc6 => constants.%.be5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.1da
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.484
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.a76
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.3f0) {
// CHECK:STDOUT:   %T.loc5_14.1 => constants.%Copy.facet.3f0
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.6b6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.fcb
// CHECK:STDOUT:   %.loc5_37.2 => constants.%.8fa
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.bf9
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.c6a
// CHECK:STDOUT:   %.loc6 => constants.%.a68
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.65f
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.c19
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
