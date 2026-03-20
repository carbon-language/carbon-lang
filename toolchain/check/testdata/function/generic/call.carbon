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
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.023 = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdc: <specific function> = specific_impl_function %impl.elem0.594, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.a87: <specific function> = specific_function %Function, @Function(%T.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.c25: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.919: <specific function> = specific_function %Function, @Function(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %.de8: Core.Form = init_form %ptr.31e [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2c7: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.411: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ed9: %ptr.as.Copy.impl.Op.type.411 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.17a: <witness> = complete_type_witness %ptr.31e [concrete]
// CHECK:STDOUT:   %Copy.facet.a7f: %Copy.type = facet_value %ptr.31e, (%Copy.impl_witness.2c7) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.9da: <specific function> = specific_function %Function, @Function(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet.c25 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.259: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %.64b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.259, %Copy.facet.a7f [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ed9, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc18.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc18: %Copy.type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc18: type = facet_access_type %T.ref.loc18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc18.3: type = converted %T.ref.loc18, %T.as_type.loc18 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc18.4: Core.Form = init_form %.loc18.3 [symbolic = %.loc18.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc10: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @Function.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc15.1: type = splice_block %.loc15.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc15: %Copy.type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc15: type = facet_access_type %T.ref.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc15.2: type = converted %T.ref.loc15, %T.as_type.loc15 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc7.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc18.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc18.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc21: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc7.1 [symbolic = %.loc21 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc21.2: @Function.%.loc21 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc21.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc21.2: <specific function> = specific_impl_function %impl.elem0.loc21.2, @Copy.WithSelf.Op(%T.loc7.1) [symbolic = %specific_impl_fn.loc21.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @Function.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc21.1: @Function.%.loc21 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc21.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc21.1: <bound method> = bound_method %x.ref, %impl.elem0.loc21.1
// CHECK:STDOUT:     %specific_impl_fn.loc21.1: <specific function> = specific_impl_function %impl.elem0.loc21.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc21.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc21.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc21.1
// CHECK:STDOUT:     %.loc18.1: ref @Function.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Function.%T.binding.as_type (%T.binding.as_type) to %.loc18.1 = call %bound_method.loc21.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc27.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc41.2: <specific function> = specific_function constants.%Function, @Function(%T.loc27.1) [symbolic = %Function.specific_fn.loc41.2 (constants.%Function.specific_fn.a87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @CallGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc43: %Copy.type = name_ref T, %T.loc27.2 [symbolic = %T.loc27.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc46: %Copy.type = converted constants.%T.binding.as_type, constants.%T.035 [symbolic = %T.loc27.1 (constants.%T.035)]
// CHECK:STDOUT:     %Function.specific_fn.loc41.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.035) [symbolic = %Function.specific_fn.loc41.2 (constants.%Function.specific_fn.a87)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc38.1 = call %Function.specific_fn.loc41.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc52.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc72.3: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc52.1) [symbolic = %.loc72.3 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc59.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc72.3: %Copy.type = facet_value %ptr.loc59.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc72.3 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:   %Function.specific_fn.loc66.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc72.3) [symbolic = %Function.specific_fn.loc66.2 (constants.%Function.specific_fn.919)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc59.1 (%ptr.e8f)) -> out %return.param: @CallGenericPtr.%ptr.loc59.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %T.ref.loc68: type = name_ref T, %T.loc52.2 [symbolic = %T.loc52.1 (constants.%T.67d)]
// CHECK:STDOUT:     %ptr.loc69: type = ptr_type %T.ref.loc68 [symbolic = %ptr.loc59.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc59.1 (%ptr.e8f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc72.1: %Copy.type = facet_value %ptr.loc69, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc72.3 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %.loc72.1: %Copy.type = converted %ptr.loc69, %Copy.facet.loc72.1 [symbolic = %Copy.facet.loc72.3 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %Copy.facet.loc72.2: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc72.3 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %.loc72.2: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc72.2 [symbolic = %Copy.facet.loc72.3 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %Function.specific_fn.loc66.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.c25) [symbolic = %Function.specific_fn.loc66.2 (constants.%Function.specific_fn.919)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc59.1 (%ptr.e8f) = call %Function.specific_fn.loc66.1(%x.ref)
// CHECK:STDOUT:     return %Function.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.31e) -> out %return.param: %ptr.31e {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %C.ref.loc94: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:   %ptr.loc95: type = ptr_type %C.ref.loc94 [concrete = constants.%ptr.31e]
// CHECK:STDOUT:   %x.ref: %ptr.31e = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet.loc98.1: %Copy.type = facet_value %ptr.loc95, (constants.%Copy.impl_witness.2c7) [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %.loc98.1: %Copy.type = converted %ptr.loc95, %Copy.facet.loc98.1 [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %Copy.facet.loc98.2: %Copy.type = facet_value constants.%ptr.31e, (constants.%Copy.impl_witness.2c7) [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %.loc98.2: %Copy.type = converted constants.%ptr.31e, %Copy.facet.loc98.2 [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.a7f) [concrete = constants.%Function.specific_fn.9da]
// CHECK:STDOUT:   %Function.call: init %ptr.31e = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.035) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc18.2 => constants.%.076a48.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67c
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.58d
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.735e75.2
// CHECK:STDOUT:   %.loc21 => constants.%.023
// CHECK:STDOUT:   %impl.elem0.loc21.2 => constants.%impl.elem0.594
// CHECK:STDOUT:   %specific_impl_fn.loc21.2 => constants.%specific_impl_fn.bdc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc27.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc38.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc52.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc59.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc63.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.c25) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%Copy.facet.c25
// CHECK:STDOUT:   %T.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc18.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d82
// CHECK:STDOUT:   %.loc21 => constants.%.299
// CHECK:STDOUT:   %impl.elem0.loc21.2 => constants.%impl.elem0.1c7
// CHECK:STDOUT:   %specific_impl_fn.loc21.2 => constants.%specific_impl_fn.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.a7f) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%Copy.facet.a7f
// CHECK:STDOUT:   %T.binding.as_type => constants.%ptr.31e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.506
// CHECK:STDOUT:   %.loc18.2 => constants.%.de8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.17a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.2c7
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.259
// CHECK:STDOUT:   %.loc21 => constants.%.64b
// CHECK:STDOUT:   %impl.elem0.loc21.2 => constants.%ptr.as.Copy.impl.Op.ed9
// CHECK:STDOUT:   %specific_impl_fn.loc21.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- deduced.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %pattern_type.ce2: type = pattern_type %Copy.type [concrete]
// CHECK:STDOUT:   %T.035: %Copy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.035 [symbolic]
// CHECK:STDOUT:   %pattern_type.9b9f0c.2: type = pattern_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %.076a48.2: Core.Form = init_form %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Function.type: type = fn_type @Function [concrete]
// CHECK:STDOUT:   %Function: %Function.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.67c: <witness> = require_complete_type %T.binding.as_type [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.58d: <witness> = lookup_impl_witness %T.035, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.735e75.2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.035) [symbolic]
// CHECK:STDOUT:   %.023: type = fn_type_with_self_type %Copy.WithSelf.Op.type.735e75.2, %T.035 [symbolic]
// CHECK:STDOUT:   %impl.elem0.594: %.023 = impl_witness_access %Copy.lookup_impl_witness.58d, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.bdc: <specific function> = specific_impl_function %impl.elem0.594, @Copy.WithSelf.Op(%T.035) [symbolic]
// CHECK:STDOUT:   %T.67d: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T.67d [symbolic]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.2d4: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.74e: %ptr.as.Copy.impl.Op.type.2d4 = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.cb6: Core.Form = init_form %ptr.e8f [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.a87: <specific function> = specific_function %Function, @Function(%T.035) [symbolic]
// CHECK:STDOUT:   %.2f2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.67d) [symbolic]
// CHECK:STDOUT:   %Copy.lookup_impl_witness.2e6: <witness> = lookup_impl_witness %ptr.e8f, @Copy [symbolic]
// CHECK:STDOUT:   %Copy.facet.c25: %Copy.type = facet_value %ptr.e8f, (%Copy.lookup_impl_witness.2e6) [symbolic]
// CHECK:STDOUT:   %Function.specific_fn.919: <specific function> = specific_function %Function, @Function(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr.31e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.506: type = pattern_type %ptr.31e [concrete]
// CHECK:STDOUT:   %.de8: Core.Form = init_form %ptr.31e [concrete]
// CHECK:STDOUT:   %Copy.impl_witness.2c7: <witness> = impl_witness imports.%Copy.impl_witness_table.c3a, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.type.411: type = fn_type @ptr.as.Copy.impl.Op, @ptr.as.Copy.impl(%C) [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.ed9: %ptr.as.Copy.impl.Op.type.411 = struct_value () [concrete]
// CHECK:STDOUT:   %complete_type.17a: <witness> = complete_type_witness %ptr.31e [concrete]
// CHECK:STDOUT:   %Copy.facet.a7f: %Copy.type = facet_value %ptr.31e, (%Copy.impl_witness.2c7) [concrete]
// CHECK:STDOUT:   %Function.specific_fn.9da: <specific function> = specific_function %Function, @Function(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.d82: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %.299: type = fn_type_with_self_type %Copy.WithSelf.Op.type.d82, %Copy.facet.c25 [symbolic]
// CHECK:STDOUT:   %impl.elem0.1c7: %.299 = impl_witness_access %Copy.lookup_impl_witness.2e6, element0 [symbolic]
// CHECK:STDOUT:   %specific_impl_fn.366: <specific function> = specific_impl_function %impl.elem0.1c7, @Copy.WithSelf.Op(%Copy.facet.c25) [symbolic]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.259: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet.a7f) [concrete]
// CHECK:STDOUT:   %.64b: type = fn_type_with_self_type %Copy.WithSelf.Op.type.259, %Copy.facet.a7f [concrete]
// CHECK:STDOUT:   %ptr.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %ptr.as.Copy.impl.Op.ed9, @ptr.as.Copy.impl.Op(%C) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.203: @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op.type (%ptr.as.Copy.impl.Op.type.2d4) = import_ref Core//prelude/parts/copy, loc217, loaded [symbolic = @ptr.as.Copy.impl.%ptr.as.Copy.impl.Op (constants.%ptr.as.Copy.impl.Op.74e)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c3a = impl_witness_table (%Core.import_ref.203), @ptr.as.Copy.impl [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Function.decl: %Function.type = fn_decl @Function [concrete = constants.%Function] {
// CHECK:STDOUT:     %T.patt: %pattern_type.ce2 = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @Function.%pattern_type (%pattern_type.9b9f0c.2) = return_slot_pattern %return.param_patt, %.loc19.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc19: %Copy.type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:     %T.as_type.loc19: type = facet_access_type %T.ref.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc19.3: type = converted %T.ref.loc19, %T.as_type.loc19 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     %.loc19.4: Core.Form = init_form %.loc19.3 [symbolic = %.loc19.2 (constants.%.076a48.2)]
// CHECK:STDOUT:     %.loc10: type = splice_block %Copy.ref [concrete = constants.%Copy.type] {
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:       %Core.ref: <namespace> = name_ref Core, imports.%Core [concrete = imports.%Core]
// CHECK:STDOUT:       %Copy.ref: type = name_ref Copy, imports.%Core.Copy [concrete = constants.%Copy.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7.2: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:     %x.param: @Function.%T.binding.as_type (%T.binding.as_type) = value_param call_param0
// CHECK:STDOUT:     %.loc16.1: type = splice_block %.loc16.2 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)] {
// CHECK:STDOUT:       %T.ref.loc16: %Copy.type = name_ref T, %T.loc7.2 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:       %T.as_type.loc16: type = facet_access_type %T.ref.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:       %.loc16.2: type = converted %T.ref.loc16, %T.as_type.loc16 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @Function.%T.binding.as_type (%T.binding.as_type) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref @Function.%T.binding.as_type (%T.binding.as_type) = out_param call_param1
// CHECK:STDOUT:     %return: ref @Function.%T.binding.as_type (%T.binding.as_type) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Function(%T.loc7.2: %Copy.type) {
// CHECK:STDOUT:   %T.loc7.1: %Copy.type = symbolic_binding T, 0 [symbolic = %T.loc7.1 (constants.%T.035)]
// CHECK:STDOUT:   %T.binding.as_type: type = symbolic_binding_type T, 0, %T.loc7.1 [symbolic = %T.binding.as_type (constants.%T.binding.as_type)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.binding.as_type [symbolic = %pattern_type (constants.%pattern_type.9b9f0c.2)]
// CHECK:STDOUT:   %.loc19.2: Core.Form = init_form %T.binding.as_type [symbolic = %.loc19.2 (constants.%.076a48.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.binding.as_type [symbolic = %require_complete (constants.%require_complete.67c)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %T.loc7.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.58d)]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%T.loc7.1) [symbolic = %Copy.WithSelf.Op.type (constants.%Copy.WithSelf.Op.type.735e75.2)]
// CHECK:STDOUT:   %.loc22: type = fn_type_with_self_type %Copy.WithSelf.Op.type, %T.loc7.1 [symbolic = %.loc22 (constants.%.023)]
// CHECK:STDOUT:   %impl.elem0.loc22.2: @Function.%.loc22 (%.023) = impl_witness_access %Copy.lookup_impl_witness, element0 [symbolic = %impl.elem0.loc22.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:   %specific_impl_fn.loc22.2: <specific function> = specific_impl_function %impl.elem0.loc22.2, @Copy.WithSelf.Op(%T.loc7.1) [symbolic = %specific_impl_fn.loc22.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @Function.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @Function.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %x.ref: @Function.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %impl.elem0.loc22.1: @Function.%.loc22 (%.023) = impl_witness_access constants.%Copy.lookup_impl_witness.58d, element0 [symbolic = %impl.elem0.loc22.2 (constants.%impl.elem0.594)]
// CHECK:STDOUT:     %bound_method.loc22.1: <bound method> = bound_method %x.ref, %impl.elem0.loc22.1
// CHECK:STDOUT:     %specific_impl_fn.loc22.1: <specific function> = specific_impl_function %impl.elem0.loc22.1, @Copy.WithSelf.Op(constants.%T.035) [symbolic = %specific_impl_fn.loc22.2 (constants.%specific_impl_fn.bdc)]
// CHECK:STDOUT:     %bound_method.loc22.2: <bound method> = bound_method %x.ref, %specific_impl_fn.loc22.1
// CHECK:STDOUT:     %.loc19.1: ref @Function.%T.binding.as_type (%T.binding.as_type) = splice_block %return.param {}
// CHECK:STDOUT:     %Copy.WithSelf.Op.call: init @Function.%T.binding.as_type (%T.binding.as_type) to %.loc19.1 = call %bound_method.loc22.2(%x.ref)
// CHECK:STDOUT:     return %Copy.WithSelf.Op.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGeneric(%T.loc28.2: %Copy.type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Function.specific_fn.loc42.2: <specific function> = specific_function constants.%Function, @Function(%T.loc28.1) [symbolic = %Function.specific_fn.loc42.2 (constants.%Function.specific_fn.a87)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGeneric.%T.binding.as_type (%T.binding.as_type)) -> out %return.param: @CallGeneric.%T.binding.as_type (%T.binding.as_type) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGeneric.%T.binding.as_type (%T.binding.as_type) = name_ref x, %x
// CHECK:STDOUT:     %.loc45: %Copy.type = converted constants.%T.binding.as_type, constants.%T.035 [symbolic = %T.loc28.1 (constants.%T.035)]
// CHECK:STDOUT:     %Function.specific_fn.loc42.1: <specific function> = specific_function %Function.ref, @Function(constants.%T.035) [symbolic = %Function.specific_fn.loc42.2 (constants.%Function.specific_fn.a87)]
// CHECK:STDOUT:     <elided>
// CHECK:STDOUT:     %Function.call: init @CallGeneric.%T.binding.as_type (%T.binding.as_type) to %.loc39.1 = call %Function.specific_fn.loc42.1(%x.ref)
// CHECK:STDOUT:     return %Function.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @CallGenericPtr(%T.loc51.2: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %.loc68.2: require_specific_def_type = require_specific_def @ptr.as.Copy.impl(%T.loc51.1) [symbolic = %.loc68.2 (constants.%.2f2)]
// CHECK:STDOUT:   %Copy.lookup_impl_witness: <witness> = lookup_impl_witness %ptr.loc58.1, @Copy [symbolic = %Copy.lookup_impl_witness (constants.%Copy.lookup_impl_witness.2e6)]
// CHECK:STDOUT:   %Copy.facet.loc68.2: %Copy.type = facet_value %ptr.loc58.1, (%Copy.lookup_impl_witness) [symbolic = %Copy.facet.loc68.2 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:   %Function.specific_fn.loc65.2: <specific function> = specific_function constants.%Function, @Function(%Copy.facet.loc68.2) [symbolic = %Function.specific_fn.loc65.2 (constants.%Function.specific_fn.919)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @CallGenericPtr.%ptr.loc58.1 (%ptr.e8f)) -> out %return.param: @CallGenericPtr.%ptr.loc58.1 (%ptr.e8f) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:     %x.ref: @CallGenericPtr.%ptr.loc58.1 (%ptr.e8f) = name_ref x, %x
// CHECK:STDOUT:     %Copy.facet.loc68.1: %Copy.type = facet_value constants.%ptr.e8f, (constants.%Copy.lookup_impl_witness.2e6) [symbolic = %Copy.facet.loc68.2 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %.loc68.1: %Copy.type = converted constants.%ptr.e8f, %Copy.facet.loc68.1 [symbolic = %Copy.facet.loc68.2 (constants.%Copy.facet.c25)]
// CHECK:STDOUT:     %Function.specific_fn.loc65.1: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.c25) [symbolic = %Function.specific_fn.loc65.2 (constants.%Function.specific_fn.919)]
// CHECK:STDOUT:     %Function.call: init @CallGenericPtr.%ptr.loc58.1 (%ptr.e8f) = call %Function.specific_fn.loc65.1(%x.ref)
// CHECK:STDOUT:     return %Function.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @CallSpecific(%x.param: %ptr.31e) -> out %return.param: %ptr.31e {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Function.ref: %Function.type = name_ref Function, file.%Function.decl [concrete = constants.%Function]
// CHECK:STDOUT:   %x.ref: %ptr.31e = name_ref x, %x
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value constants.%ptr.31e, (constants.%Copy.impl_witness.2c7) [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %.loc91: %Copy.type = converted constants.%ptr.31e, %Copy.facet [concrete = constants.%Copy.facet.a7f]
// CHECK:STDOUT:   %Function.specific_fn: <specific function> = specific_function %Function.ref, @Function(constants.%Copy.facet.a7f) [concrete = constants.%Function.specific_fn.9da]
// CHECK:STDOUT:   %Function.call: init %ptr.31e = call %Function.specific_fn(%x.ref)
// CHECK:STDOUT:   return %Function.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%T.035) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc19.2 => constants.%.076a48.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.67c
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.58d
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.735e75.2
// CHECK:STDOUT:   %.loc22 => constants.%.023
// CHECK:STDOUT:   %impl.elem0.loc22.2 => constants.%impl.elem0.594
// CHECK:STDOUT:   %specific_impl_fn.loc22.2 => constants.%specific_impl_fn.bdc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGeneric(constants.%T.035) {
// CHECK:STDOUT:   %T.loc28.1 => constants.%T.035
// CHECK:STDOUT:   %T.binding.as_type => constants.%T.binding.as_type
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.9b9f0c.2
// CHECK:STDOUT:   %.loc39.2 => constants.%.076a48.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @CallGenericPtr(constants.%T.67d) {
// CHECK:STDOUT:   %T.loc51.1 => constants.%T.67d
// CHECK:STDOUT:   %ptr.loc58.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc62.1 => constants.%.cb6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.c25) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%Copy.facet.c25
// CHECK:STDOUT:   %T.binding.as_type => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc19.2 => constants.%.cb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete.ef1
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.lookup_impl_witness.2e6
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.d82
// CHECK:STDOUT:   %.loc22 => constants.%.299
// CHECK:STDOUT:   %impl.elem0.loc22.2 => constants.%impl.elem0.1c7
// CHECK:STDOUT:   %specific_impl_fn.loc22.2 => constants.%specific_impl_fn.366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Function(constants.%Copy.facet.a7f) {
// CHECK:STDOUT:   %T.loc7.1 => constants.%Copy.facet.a7f
// CHECK:STDOUT:   %T.binding.as_type => constants.%ptr.31e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.506
// CHECK:STDOUT:   %.loc19.2 => constants.%.de8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.17a
// CHECK:STDOUT:   %Copy.lookup_impl_witness => constants.%Copy.impl_witness.2c7
// CHECK:STDOUT:   %Copy.WithSelf.Op.type => constants.%Copy.WithSelf.Op.type.259
// CHECK:STDOUT:   %.loc22 => constants.%.64b
// CHECK:STDOUT:   %impl.elem0.loc22.2 => constants.%ptr.as.Copy.impl.Op.ed9
// CHECK:STDOUT:   %specific_impl_fn.loc22.2 => constants.%ptr.as.Copy.impl.Op.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
