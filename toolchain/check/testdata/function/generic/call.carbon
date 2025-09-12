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
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.e31: <specific function> = specific_function %Function, @Function(%T.578) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.def: %Copy.type = facet_value %ptr.79f, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.329: <specific function> = specific_function %Function, @Function(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a94: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0f5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fd1: %ptr.as.Copy.impl.Op.type.0f5 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.d05: <witness> = complete_type_witness %ptr.019 [concrete]
// CHECK:STDOUT:   %Copy.facet.9a6: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.a94) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.527: <specific function> = specific_function %Function, @Function(%Copy.facet.9a6) [concrete]
// CHECK:STDOUT:   %.fe5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.def [symbolic]
// CHECK:STDOUT:   %impl.elem0.429: %.fe5 = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d8a: <specific function> = specific_impl_function %impl.elem0.429, @Copy.Op(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %.a98: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9a6 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fd1, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_37: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc5_37: type = facet_access_type %T.ref.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37: type = converted %T.ref.loc5_37, %T.as_type.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_21: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:     %x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_31.1: type = splice_block %.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_31: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc5_31.2: type = facet_access_type %T.ref.loc5_31 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_31.2: type = converted %T.ref.loc5_31, %T.as_type.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.as_type.loc5_31.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:   %T.as_type.loc5_31.1: type = facet_access_type %T.loc5_13.1 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_31.1 [symbolic = %pattern_type (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_31.1 [symbolic = %require_complete (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_13.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc5_31.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc6_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_10.2 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @Function.%.loc6_10.2 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type)) -> @Function.%T.as_type.loc5_31.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.as_type.loc5_31.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @Function.%.loc6_10.2 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %x.ref, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc6_10.1: init @Function.%T.as_type.loc5_31.1 (%T.as_type) = call %bound_method.loc6_10.2(%x.ref)
// CHECK:STDOUT:     return %.loc6_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc10_16.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc12_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc10_16.1) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.e31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type)) -> @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc12: %Copy.type = name_ref T, %T.loc10_16.2 [symbolic = %T.loc10_16.1 (constants.%T.578)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc12_23.1: %Copy.type = converted constants.%T.as_type, constants.%T.578 [symbolic = %T.loc10_16.1 (constants.%T.578)]
// CHECK:STDOUT:     %.loc12_23.2: %Copy.type = converted constants.%T.as_type, constants.%T.578 [symbolic = %T.loc10_16.1 (constants.%T.578)]
// CHECK:STDOUT:     %Function.specific_fn.loc12_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.578) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.e31)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = call %Function.specific_fn.loc12_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc16_19.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_33.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet.loc18_24.4: %Copy.type = facet_value %ptr.loc16_33.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:   %Function.specific_fn.loc18_10.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc18_24.4) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.329)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc18: type = name_ref T, %T.loc16_19.2 [symbolic = %T.loc16_19.1 (constants.%T.8b3)]
// CHECK:STDOUT:     %ptr.loc18: type = ptr_type %T.ref.loc18 [symbolic = %ptr.loc16_33.1 (constants.%ptr.79f)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc18_24.1: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %.loc18_24.1: %Copy.type = converted %ptr.loc18, %Copy.facet.loc18_24.1 [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %Copy.facet.loc18_24.2: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %.loc18_24.2: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc18_24.2 [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %Copy.facet.loc18_24.3: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %.loc18_24.3: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc18_24.3 [symbolic = %Copy.facet.loc18_24.4 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %Function.specific_fn.loc18_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.def) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.329)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) = call %Function.specific_fn.loc18_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.019) -> %ptr.019 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc26: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc26: type = ptr_type %C.ref.loc26 [concrete = constants.%ptr.019]
// CHECK:STDOUT:   %x.ref: %ptr.019 = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet.loc26_24.1: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc26_24.1: %Copy.type = converted %ptr.loc26, %Copy.facet.loc26_24.1 [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %Copy.facet.loc26_24.2: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc26_24.2: %Copy.type = converted constants.%ptr.019, %Copy.facet.loc26_24.2 [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %Copy.facet.loc26_24.3: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc26_24.3: %Copy.type = converted constants.%ptr.019, %Copy.facet.loc26_24.3 [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.9a6) [concrete = constants.%Function.specific_fn.527]
// CHECK:STDOUT:   %Function.call: init %ptr.019 = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.578) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ecc
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.3ba
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.72e
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.671
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.56e
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.fb7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.578) {
// CHECK:STDOUT:   %T.loc10_16.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc10_34.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc16_19.1 => constants.%T.8b3
// CHECK:STDOUT:   %ptr.loc16_33.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.def) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Copy.facet.def
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.cb2
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.def
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.fe5
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.429
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.d8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.9a6) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Copy.facet.9a6
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.019
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.44a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d05
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.a94
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.9a6
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.a98
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.fd1
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %T.578: %Copy.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.as_type: type = facet_access_type %T.578 [symbolic]
// CHECK:STDOUT:   %pattern_type.f8cebc.1: type = pattern_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ecc: <witness> = require_complete_type %T.as_type [symbolic]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.3ba: <witness> = lookup_impl_witness %T.578, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.72e: %Copy.type = facet_value %T.as_type, (%Copy.lookup_impl_witness.3ba) [symbolic]
// CHECK:STDOUT:   %.671: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.72e [symbolic]
// CHECK:STDOUT:   %impl.elem0.56e: %.671 = impl_witness_access %Copy.lookup_impl_witness.3ba, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.fb7: <specific function> = specific_impl_function %impl.elem0.56e, @Copy.Op(%Copy.facet.72e) [symbolic]
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.79f: type = ptr_type %T.8b3 [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.f23: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.8b3) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.abf: %ptr.as.Copy.impl.Op.type.f23 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.afe: type = pattern_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %require_complete.6e5: <witness> = require_complete_type %ptr.79f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.e31: <specific function> = specific_function %Function, @Function(%T.578) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.cb2: <witness> = lookup_impl_witness %ptr.79f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.def: %Copy.type = facet_value %ptr.79f, (%Copy.lookup_impl_witness.cb2) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.329: <specific function> = specific_function %Function, @Function(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %ptr.019: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.44a: type = pattern_type %ptr.019 [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.a94: <witness> = impl_witness imports.%Copy.impl_witness_table.a71, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.0f5: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.fd1: %ptr.as.Copy.impl.Op.type.0f5 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.d05: <witness> = complete_type_witness %ptr.019 [concrete]
// CHECK:STDOUT:   %Copy.facet.9a6: %Copy.type = facet_value %ptr.019, (%Copy.impl_witness.a94) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.527: <specific function> = specific_function %Function, @Function(%Copy.facet.9a6) [concrete]
// CHECK:STDOUT:   %.fe5: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.def [symbolic]
// CHECK:STDOUT:   %impl.elem0.429: %.fe5 = impl_witness_access %Copy.lookup_impl_witness.cb2, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.d8a: <specific function> = specific_impl_function %impl.elem0.429, @Copy.Op(%Copy.facet.def) [symbolic]
// CHECK:STDOUT:   %.a98: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet.9a6 [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.fd1, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.de9: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.f23) = import_ref Core//prelude/parts/copy, loc36_31, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.abf)]
// CHECK:STDOUT:   %Copy.impl_witness_table.a71 = impl_witness_table (%Core.import_ref.de9), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.f8cebc.1) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc5_37: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:     %T.as_type.loc5_37: type = facet_access_type %T.ref.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_37: type = converted %T.ref.loc5_37, %T.as_type.loc5_37 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     %.loc5_21: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc5_13.2: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:     %x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc5_31.1: type = splice_block %.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)] {
// CHECK:STDOUT:       %T.ref.loc5_31: %Copy.type = name_ref T, %T.loc5_13.2 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:       %T.as_type.loc5_31.2: type = facet_access_type %T.ref.loc5_31 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:       %.loc5_31.2: type = converted %T.ref.loc5_31, %T.as_type.loc5_31.2 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.as_type.loc5_31.1 (%T.as_type) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.as_type.loc5_31.1 (%T.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc5_13.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc5_13.1: %Copy.type = bind_symbolic_name T, 0 [symbolic = %T.loc5_13.1 (constants.%T.578)]
// CHECK:STDOUT:   %T.as_type.loc5_31.1: type = facet_access_type %T.loc5_13.1 [symbolic = %T.as_type.loc5_31.1 (constants.%T.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.as_type.loc5_31.1 [symbolic = %pattern_type (constants.%pattern_type.f8cebc.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.as_type.loc5_31.1 [symbolic = %require_complete (constants.%require_complete.ecc)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc5_13.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.3ba)]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %T.as_type.loc5_31.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet (constants.%Copy.facet.72e)]
// CHECK:STDOUT:   %.loc6_10.2: type = fn_type_with_self_type constants.%Copy.Op.type, %Copy.facet [symbolic = %.loc6_10.2 (constants.%.671)]
// CHECK:STDOUT:   %impl.elem0.loc6_10.2: @Function.%.loc6_10.2 (%.671) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2: <specific function> = specific_impl_function %impl.elem0.loc6_10.2, @Copy.Op(%Copy.facet) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.as_type.loc5_31.1 (%T.as_type)) -> @Function.%T.as_type.loc5_31.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.as_type.loc5_31.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc6_10.1: @Function.%.loc6_10.2 (%.671) = impl_witness_access constants.%Copy.lookup_impl_witness.3ba, element0 [symbolic = %impl.elem0.loc6_10.2 (constants.%impl.elem0.56e)]
// CHECK:STDOUT:     %bound_method.loc6_10.1: <bound method> = bound_method %x.ref, %impl.elem0.loc6_10.1
// CHECK:STDOUT:     %specific_impl_fn.loc6_10.1: <specific function> = specific_impl_function %impl.elem0.loc6_10.1, @Copy.Op(constants.%Copy.facet.72e) [symbolic = %specific_impl_fn.loc6_10.2 (constants.%specific_impl_fn.fb7)]
// CHECK:STDOUT:     %bound_method.loc6_10.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc6_10.1
// CHECK:STDOUT:     %.loc6_10.1: init @Function.%T.as_type.loc5_31.1 (%T.as_type) = call %bound_method.loc6_10.2(%x.ref)
// CHECK:STDOUT:     return %.loc6_10.1 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc10_16.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc12_10.2: <specific function> = specific_function constants.%Function, @Function(%T.loc10_16.1) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.e31)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type)) -> @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc12_20.1: %Copy.type = converted constants.%T.as_type, constants.%T.578 [symbolic = %T.loc10_16.1 (constants.%T.578)]
// CHECK:STDOUT:     %.loc12_20.2: %Copy.type = converted constants.%T.as_type, constants.%T.578 [symbolic = %T.loc10_16.1 (constants.%T.578)]
// CHECK:STDOUT:     %Function.specific_fn.loc12_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.578) [symbolic = %Function.specific_fn.loc12_10.2 (constants.%Function.specific_fn.e31)]
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.as_type.loc10_34.1 (%T.as_type) = call %Function.specific_fn.loc12_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc16_19.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc16_33.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.cb2)]
// CHECK:STDOUT:   %Copy.facet.loc18_20.3: %Copy.type = facet_value %ptr.loc16_33.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc18_20.3 (constants.%Copy.facet.def)]
// CHECK:STDOUT:   %Function.specific_fn.loc18_10.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc18_20.3) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.329)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f)) -> @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc18_20.1: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc18_20.3 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %.loc18_20.1: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc18_20.1 [symbolic = %Copy.facet.loc18_20.3 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %Copy.facet.loc18_20.2: %Copy.type = facet_value constants.%ptr.79f, (constants.%Copy.lookup_impl_witness.cb2) [symbolic = %Copy.facet.loc18_20.3 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %.loc18_20.2: %Copy.type = converted constants.%ptr.79f, %Copy.facet.loc18_20.2 [symbolic = %Copy.facet.loc18_20.3 (constants.%Copy.facet.def)]
// CHECK:STDOUT:     %Function.specific_fn.loc18_10.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.def) [symbolic = %Function.specific_fn.loc18_10.2 (constants.%Function.specific_fn.329)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc16_33.1 (%ptr.79f) = call %Function.specific_fn.loc18_10.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.019) -> %ptr.019 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %x.ref: %ptr.019 = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet.loc26_20.1: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc26_20.1: %Copy.type = converted constants.%ptr.019, %Copy.facet.loc26_20.1 [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %Copy.facet.loc26_20.2: %Copy.type = facet_value constants.%ptr.019, (constants.%Copy.impl_witness.a94) [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %.loc26_20.2: %Copy.type = converted constants.%ptr.019, %Copy.facet.loc26_20.2 [concrete = constants.%Copy.facet.9a6]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.9a6) [concrete = constants.%Function.specific_fn.527]
// CHECK:STDOUT:   %Function.call: init %ptr.019 = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.578) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ecc
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.3ba
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.72e
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.671
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.56e
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.fb7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.578) {
// CHECK:STDOUT:   %T.loc10_16.1 => constants.%T.578
// CHECK:STDOUT:   %T.as_type.loc10_34.1 => constants.%T.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f8cebc.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc16_19.1 => constants.%T.8b3
// CHECK:STDOUT:   %ptr.loc16_33.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.def) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Copy.facet.def
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.79f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.afe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.6e5
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.cb2
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.def
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.fe5
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%impl.elem0.429
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%specific_impl_fn.d8a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.9a6) {
// CHECK:STDOUT:   %T.loc5_13.1 => constants.%Copy.facet.9a6
// CHECK:STDOUT:   %T.as_type.loc5_31.1 => constants.%ptr.019
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.44a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.d05
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.a94
// CHECK:STDOUT:   %Copy.facet => constants.%Copy.facet.9a6
// CHECK:STDOUT:   %.loc6_10.2 => constants.%.a98
// CHECK:STDOUT:   %impl.elem0.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.fd1
// CHECK:STDOUT:   %specific_impl_fn.loc6_10.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
