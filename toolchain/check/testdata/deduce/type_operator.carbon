// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/type_operator.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/type_operator.carbon

// --- pointer.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: C*) -> C {
  return F(p);
}

// --- const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: const C*) -> C {
  return F(p);
}

// --- nonconst_from_const.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: T*) -> T { return F(p); }

fn G(p: const C*) -> const C {
  return F(p);
}

// --- fail_const_from_nonconst.carbon

library "[[@TEST_NAME]]";

class C {}

fn F[T:! type](p: const T*) -> T { return F(p); }

fn G(p: C*) -> const C {
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE+7]]:10: error: cannot deduce value for generic parameter `T` [DeductionIncomplete]
  // CHECK:STDERR:   return F(p);
  // CHECK:STDERR:          ^~~~
  // CHECK:STDERR: fail_const_from_nonconst.carbon:[[@LINE-6]]:1: note: while deducing parameters of generic declared here [DeductionGenericHere]
  // CHECK:STDERR: fn F[T:! type](p: const T*) -> T { return F(p); }
  // CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  return F(p);
}

// CHECK:STDOUT: --- pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.1b6: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %ptr.f2f: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.d9b: type = pattern_type %ptr.f2f [concrete]
// CHECK:STDOUT:   %.2d5: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.bd2: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ab1: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.d74: <witness> = complete_type_witness %ptr.f2f [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_17 (%pattern_type.4f4) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc6_26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_26.3: Core.Form = init_form %T.ref.loc6_26 [symbolic = %.loc6_26.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.1 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.2 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.2: type = ptr_type %T.ref.loc6_19 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.1 (%ptr.e8f) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.d9b = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.d9b = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.bd2 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.bd2 = return_slot_pattern %return.param_patt, %C.ref.loc8_16 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_16: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_16.2: Core.Form = init_form %C.ref.loc8_16 [concrete = constants.%.2d5]
// CHECK:STDOUT:     %p.param: %ptr.f2f = value_param call_param0
// CHECK:STDOUT:     %.loc8_10: type = splice_block %ptr [concrete = constants.%ptr.f2f] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc8_9 [concrete = constants.%ptr.f2f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.f2f = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_20.1: type = ptr_type %T.loc6_7.1 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %ptr.loc6_20.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %.loc6_26.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_26: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_20.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %T.loc6_7.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.1b6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_20.1 (%ptr.e8f)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.1 (%ptr.e8f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.1b6)]
// CHECK:STDOUT:     %.loc6_26.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_26.1 = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.f2f) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.f2f = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.ab1]
// CHECK:STDOUT:   %.loc8_16.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %C to %.loc8_16.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.ef1
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.1b6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.f2f
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.d9b
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.2d5
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.bd2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.d74
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.ab1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.4ff: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.a15: type = ptr_type %const.4ff [symbolic]
// CHECK:STDOUT:   %pattern_type.26f: type = pattern_type %ptr.a15 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0c1: <witness> = require_complete_type %ptr.a15 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ded: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const.763: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.10b: type = ptr_type %const.763 [concrete]
// CHECK:STDOUT:   %pattern_type.bd2: type = pattern_type %ptr.10b [concrete]
// CHECK:STDOUT:   %.3a9: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.e52: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.f37: <specific function> = specific_function %F, @F(%C) [concrete]
// CHECK:STDOUT:   %complete_type.d4b: <witness> = complete_type_witness %ptr.10b [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.26f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_17 (%pattern_type.26f) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_32 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_32 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc6_32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_32.3: Core.Form = init_form %T.ref.loc6_32 [symbolic = %.loc6_32.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.1 (%ptr.a15) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.2: type = const_type %T.ref.loc6_25 [symbolic = %const.loc6_19.1 (constants.%const.4ff)]
// CHECK:STDOUT:       %ptr.loc6_26.2: type = ptr_type %const.loc6_19.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.1 (%ptr.a15) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.bd2 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.bd2 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e52 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e52 = return_slot_pattern %return.param_patt, %C.ref.loc8_22 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc8_22.2: Core.Form = init_form %C.ref.loc8_22 [concrete = constants.%.3a9]
// CHECK:STDOUT:     %p.param: %ptr.10b = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.10b] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const: type = const_type %C.ref.loc8_15 [concrete = constants.%const.763]
// CHECK:STDOUT:       %ptr: type = ptr_type %const [concrete = constants.%ptr.10b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.10b = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %const.loc6_19.1: type = const_type %T.loc6_7.1 [symbolic = %const.loc6_19.1 (constants.%const.4ff)]
// CHECK:STDOUT:   %ptr.loc6_26.1: type = ptr_type %const.loc6_19.1 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %ptr.loc6_26.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.26f)]
// CHECK:STDOUT:   %.loc6_32.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_32.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_26.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.0c1)]
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type %T.loc6_7.1 [symbolic = %require_complete.loc6_29 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ded)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_26.1 (%ptr.a15)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.1 (%ptr.a15) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn.ded)]
// CHECK:STDOUT:     %.loc6_32.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_32.1 = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.10b) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.10b = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.f37]
// CHECK:STDOUT:   %.loc8_22.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %C to %.loc8_22.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.4ff
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.a15
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.26f
// CHECK:STDOUT:   %.loc6_32.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.0c1
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.ded
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%C
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.763
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.10b
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.bd2
// CHECK:STDOUT:   %.loc6_32.2 => constants.%.3a9
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.e52
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.d4b
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%complete_type.357
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn.f37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nonconst_from_const.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %ptr.e8f: type = ptr_type %T [symbolic]
// CHECK:STDOUT:   %pattern_type.4f4: type = pattern_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.ef1: <witness> = require_complete_type %ptr.e8f [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.5d4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %const: type = const_type %C [concrete]
// CHECK:STDOUT:   %ptr.2e1: type = ptr_type %const [concrete]
// CHECK:STDOUT:   %pattern_type.6ae: type = pattern_type %ptr.2e1 [concrete]
// CHECK:STDOUT:   %.fe6: Core.Form = init_form %const [concrete]
// CHECK:STDOUT:   %pattern_type.cb8: type = pattern_type %const [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.461: <specific function> = specific_function %F, @F(%const) [concrete]
// CHECK:STDOUT:   %complete_type.98c: <witness> = complete_type_witness %ptr.2e1 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.4f4) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_17 (%pattern_type.4f4) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_26 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_26 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc6_26 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_26: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_26.3: Core.Form = init_form %T.ref.loc6_26 [symbolic = %.loc6_26.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_20.1 (%ptr.e8f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_20: type = splice_block %ptr.loc6_20.2 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)] {
// CHECK:STDOUT:       %T.ref.loc6_19: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %ptr.loc6_20.2: type = ptr_type %T.ref.loc6_19 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_20.1 (%ptr.e8f) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.6ae = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.6ae = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb8 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb8 = return_slot_pattern %return.param_patt, %const.loc8_22 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_28: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const.loc8_22: type = const_type %C.ref.loc8_28 [concrete = constants.%const]
// CHECK:STDOUT:     %.loc8_22.2: Core.Form = init_form %const.loc8_22 [concrete = constants.%.fe6]
// CHECK:STDOUT:     %p.param: %ptr.2e1 = value_param call_param0
// CHECK:STDOUT:     %.loc8_16: type = splice_block %ptr [concrete = constants.%ptr.2e1] {
// CHECK:STDOUT:       %C.ref.loc8_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %const.loc8_9: type = const_type %C.ref.loc8_15 [concrete = constants.%const]
// CHECK:STDOUT:       %ptr: type = ptr_type %const.loc8_9 [concrete = constants.%ptr.2e1]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.2e1 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const = out_param call_param1
// CHECK:STDOUT:     %return: ref %const = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %ptr.loc6_20.1: type = ptr_type %T.loc6_7.1 [symbolic = %ptr.loc6_20.1 (constants.%ptr.e8f)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %ptr.loc6_20.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.4f4)]
// CHECK:STDOUT:   %.loc6_26.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_26.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_26: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_26 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_20.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.ef1)]
// CHECK:STDOUT:   %require_complete.loc6_23: <witness> = require_complete_type %T.loc6_7.1 [symbolic = %require_complete.loc6_23 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.5d4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_20.1 (%ptr.e8f)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_20.1 (%ptr.e8f) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_37.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_37.2 (constants.%F.specific_fn.5d4)]
// CHECK:STDOUT:     %.loc6_26.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_26.1 = call %F.specific_fn.loc6_37.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.2e1) -> out %return.param: %const {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.2e1 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%const) [concrete = constants.%F.specific_fn.461]
// CHECK:STDOUT:   %.loc8_22.1: ref %const = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %const to %.loc8_22.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.e8f
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.4f4
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.ef1
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.5d4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%const) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%const
// CHECK:STDOUT:   %ptr.loc6_20.1 => constants.%ptr.2e1
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.6ae
// CHECK:STDOUT:   %.loc6_26.2 => constants.%.fe6
// CHECK:STDOUT:   %pattern_type.loc6_26 => constants.%pattern_type.cb8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%complete_type.98c
// CHECK:STDOUT:   %require_complete.loc6_23 => constants.%complete_type.357
// CHECK:STDOUT:   %F.specific_fn.loc6_37.2 => constants.%F.specific_fn.461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_const_from_nonconst.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %const.4ff: type = const_type %T [symbolic]
// CHECK:STDOUT:   %ptr.a15: type = ptr_type %const.4ff [symbolic]
// CHECK:STDOUT:   %pattern_type.26f: type = pattern_type %ptr.a15 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.0c1: <witness> = require_complete_type %ptr.a15 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %ptr.52e: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %pattern_type.137: type = pattern_type %ptr.52e [concrete]
// CHECK:STDOUT:   %const.b0d: type = const_type %C [concrete]
// CHECK:STDOUT:   %.7da: Core.Form = init_form %const.b0d [concrete]
// CHECK:STDOUT:   %pattern_type.5e1: type = pattern_type %const.b0d [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc6_17 (%pattern_type.26f) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc6_17 (%pattern_type.26f) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc6_32 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc6_32 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc6_32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc6_32: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc6_32.3: Core.Form = init_form %T.ref.loc6_32 [symbolic = %.loc6_32.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc6_10.1: type = splice_block %.loc6_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc6_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc6_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%ptr.loc6_26.1 (%ptr.a15) = value_param call_param0
// CHECK:STDOUT:     %.loc6_26: type = splice_block %ptr.loc6_26.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)] {
// CHECK:STDOUT:       %T.ref.loc6_25: type = name_ref T, %T.loc6_7.2 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:       %const.loc6_19.2: type = const_type %T.ref.loc6_25 [symbolic = %const.loc6_19.1 (constants.%const.4ff)]
// CHECK:STDOUT:       %ptr.loc6_26.2: type = ptr_type %const.loc6_19.2 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%ptr.loc6_26.1 (%ptr.a15) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc6_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc6_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.137 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.137 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.5e1 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.5e1 = return_slot_pattern %return.param_patt, %const [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc8_22: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %const: type = const_type %C.ref.loc8_22 [concrete = constants.%const.b0d]
// CHECK:STDOUT:     %.loc8_16: Core.Form = init_form %const [concrete = constants.%.7da]
// CHECK:STDOUT:     %p.param: %ptr.52e = value_param call_param0
// CHECK:STDOUT:     %.loc8_10: type = splice_block %ptr [concrete = constants.%ptr.52e] {
// CHECK:STDOUT:       %C.ref.loc8_9: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref.loc8_9 [concrete = constants.%ptr.52e]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr.52e = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %const.b0d = out_param call_param1
// CHECK:STDOUT:     %return: ref %const.b0d = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc6_7.2: type) {
// CHECK:STDOUT:   %T.loc6_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc6_7.1 (constants.%T)]
// CHECK:STDOUT:   %const.loc6_19.1: type = const_type %T.loc6_7.1 [symbolic = %const.loc6_19.1 (constants.%const.4ff)]
// CHECK:STDOUT:   %ptr.loc6_26.1: type = ptr_type %const.loc6_19.1 [symbolic = %ptr.loc6_26.1 (constants.%ptr.a15)]
// CHECK:STDOUT:   %pattern_type.loc6_17: type = pattern_type %ptr.loc6_26.1 [symbolic = %pattern_type.loc6_17 (constants.%pattern_type.26f)]
// CHECK:STDOUT:   %.loc6_32.2: Core.Form = init_form %T.loc6_7.1 [symbolic = %.loc6_32.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc6_32: type = pattern_type %T.loc6_7.1 [symbolic = %pattern_type.loc6_32 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17: <witness> = require_complete_type %ptr.loc6_26.1 [symbolic = %require_complete.loc6_17 (constants.%require_complete.0c1)]
// CHECK:STDOUT:   %require_complete.loc6_29: <witness> = require_complete_type %T.loc6_7.1 [symbolic = %require_complete.loc6_29 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2: <specific function> = specific_function constants.%F, @F(%T.loc6_7.1) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%ptr.loc6_26.1 (%ptr.a15)) -> out %return.param: @F.%T.loc6_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%ptr.loc6_26.1 (%ptr.a15) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc6_43.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc6_43.2 (constants.%F.specific_fn)]
// CHECK:STDOUT:     %.loc6_32.1: ref @F.%T.loc6_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc6_7.1 (%T) to %.loc6_32.1 = call %F.specific_fn.loc6_43.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %ptr.52e) -> out %return.param: %const.b0d {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %ptr.52e = name_ref p, %p
// CHECK:STDOUT:   return <error> to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc6_7.1 => constants.%T
// CHECK:STDOUT:   %const.loc6_19.1 => constants.%const.4ff
// CHECK:STDOUT:   %ptr.loc6_26.1 => constants.%ptr.a15
// CHECK:STDOUT:   %pattern_type.loc6_17 => constants.%pattern_type.26f
// CHECK:STDOUT:   %.loc6_32.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc6_32 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc6_17 => constants.%require_complete.0c1
// CHECK:STDOUT:   %require_complete.loc6_29 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc6_43.2 => constants.%F.specific_fn
// CHECK:STDOUT: }
// CHECK:STDOUT:
