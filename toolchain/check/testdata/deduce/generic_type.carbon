// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/deduce/generic_type.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/deduce/generic_type.carbon

// --- class.carbon

library "[[@TEST_NAME]]";

class C(T:! type) {}
class D {}

fn F[T:! type](p: C(T)) -> T { return F(p); }

fn G(p: C(D)) -> D {
  return F(p);
}

// --- interface.carbon

library "[[@TEST_NAME]]";

class I(T:! type) {}
class C {}

fn F[T:! type](p: I(T)) -> C { return F(p); }

fn G(p: I(C)) -> C {
  return F(p);
}

// --- nested.carbon

library "[[@TEST_NAME]]";

class Outer(T:! type) {
  class Inner(U:! type) {}
}

class C {}
class D {}

// C++ doesn't permit deducing `T` here because `Outer` might be specialized.
// But that's not possible in Carbon, so we can deduce `T`.
fn F[T:! type, U:! type](p: Outer(T).Inner(U)) -> (T, U) { return F(p); }

fn G(p: Outer(C).Inner(D)) -> (C, D) {
  return F(p);
}

// --- nontype.carbon

library "[[@TEST_NAME]]";

class WithNontype(N:! i32) {}

fn F[N:! i32](unused x: WithNontype(N)) -> i32 { return N; }

fn G() -> i32 {
  return F({} as WithNontype(0));
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.1f9: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.69e: type = pattern_type %C.1f9 [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.122: <witness> = require_complete_type %C.1f9 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.d7f: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %C.11b: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %pattern_type.978: type = pattern_type %C.11b [concrete]
// CHECK:STDOUT:   %.f19: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.f1a: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.634: <specific function> = specific_function %F, @F(%D) [concrete]
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
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [concrete = constants.%C.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc7_17 (%pattern_type.69e) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc7_17 (%pattern_type.69e) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_28 (%pattern_type.51d) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_28 (%pattern_type.51d) = return_slot_pattern %return.param_patt, %T.ref.loc7_28 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_28: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_28.3: Core.Form = init_form %T.ref.loc7_28 [symbolic = %.loc7_28.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_22.1 (%C.1f9) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %C.loc7_22.2 [symbolic = %C.loc7_22.1 (constants.%C.1f9)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc7_21: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_22.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_22.1 (constants.%C.1f9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%C.loc7_22.1 (%C.1f9) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.978 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.978 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.f1a = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.f1a = return_slot_pattern %return.param_patt, %D.ref.loc9_18 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %D.ref.loc9_18 [concrete = constants.%.f19]
// CHECK:STDOUT:     %p.param: %C.11b = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %C [concrete = constants.%C.11b] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc9_11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.11b]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %C.11b = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.1f9
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type) {
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc7_22.1: type = class_type @C, @C(%T.loc7_7.1) [symbolic = %C.loc7_22.1 (constants.%C.1f9)]
// CHECK:STDOUT:   %pattern_type.loc7_17: type = pattern_type %C.loc7_22.1 [symbolic = %pattern_type.loc7_17 (constants.%pattern_type.69e)]
// CHECK:STDOUT:   %.loc7_28.2: Core.Form = init_form %T.loc7_7.1 [symbolic = %.loc7_28.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_28: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7_28 (constants.%pattern_type.51d)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_17: <witness> = require_complete_type %C.loc7_22.1 [symbolic = %require_complete.loc7_17 (constants.%require_complete.122)]
// CHECK:STDOUT:   %require_complete.loc7_25: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7_25 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.d7f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%C.loc7_22.1 (%C.1f9)) -> out %return.param: @F.%T.loc7_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_22.1 (%C.1f9) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.d7f)]
// CHECK:STDOUT:     %.loc7_28.1: ref @F.%T.loc7_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_7.1 (%T) to %.loc7_28.1 = call %F.specific_fn.loc7_39.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %C.11b) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.11b = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn.634]
// CHECK:STDOUT:   %.loc9_18.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %D to %.loc9_18.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %C.loc7_22.1 => constants.%C.1f9
// CHECK:STDOUT:   %pattern_type.loc7_17 => constants.%pattern_type.69e
// CHECK:STDOUT:   %.loc7_28.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_28 => constants.%pattern_type.51d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_17 => constants.%require_complete.122
// CHECK:STDOUT:   %require_complete.loc7_25 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.d7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%D
// CHECK:STDOUT:   %C.loc7_22.1 => constants.%C.11b
// CHECK:STDOUT:   %pattern_type.loc7_17 => constants.%pattern_type.978
// CHECK:STDOUT:   %.loc7_28.2 => constants.%.f19
// CHECK:STDOUT:   %pattern_type.loc7_28 => constants.%pattern_type.f1a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_17 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc7_25 => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.634
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_class_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.a0e: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.108: type = pattern_type %I.a0e [symbolic]
// CHECK:STDOUT:   %.8f0: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.0d5: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.a0e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.d7f: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %I.870: type = class_type @I, @I(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.ec4: type = pattern_type %I.870 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.076: <specific function> = specific_function %F, @F(%C) [concrete]
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
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type = class_decl @I [concrete = constants.%I.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_13.1: type = splice_block %.loc4_13.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_13.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type (%pattern_type.108) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type (%pattern_type.108) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0d5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0d5 = return_slot_pattern %return.param_patt, %C.ref [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc7_28.2: Core.Form = init_form %C.ref [concrete = constants.%.8f0]
// CHECK:STDOUT:     %.loc7_10.1: type = splice_block %.loc7_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc7_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_22.1 (%I.a0e) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %I.loc7_22.2 [symbolic = %I.loc7_22.1 (constants.%I.a0e)] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %I.loc7_22.2: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_22.1 (constants.%I.a0e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%I.loc7_22.1 (%I.a0e) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.ec4 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.ec4 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.0d5 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.0d5 = return_slot_pattern %return.param_patt, %C.ref.loc9_18 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %C.ref.loc9_18 [concrete = constants.%.8f0]
// CHECK:STDOUT:     %p.param: %I.870 = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I [concrete = constants.%I.870] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %C.ref.loc9_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %I: type = class_type @I, @I(constants.%C) [concrete = constants.%I.870]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %I.870 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.a0e
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
// CHECK:STDOUT: generic fn @F(%T.loc7_7.2: type) {
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %I.loc7_22.1: type = class_type @I, @I(%T.loc7_7.1) [symbolic = %I.loc7_22.1 (constants.%I.a0e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.loc7_22.1 [symbolic = %pattern_type (constants.%pattern_type.108)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.loc7_22.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.d7f)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%I.loc7_22.1 (%I.a0e)) -> out %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_22.1 (%I.a0e) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.d7f)]
// CHECK:STDOUT:     %.loc7_28.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init %C to %.loc7_28.1 = call %F.specific_fn.loc7_39.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %I.870) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.870 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.076]
// CHECK:STDOUT:   %.loc9_18.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %C to %.loc9_18.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %I.loc7_22.1 => constants.%I.a0e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.108
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.d7f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%C
// CHECK:STDOUT:   %I.loc7_22.1 => constants.%I.870
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.ec4
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.076
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.3b5: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.e77: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.92c: %Inner.type.e77 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.6c4: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %require_complete.f6c: <witness> = require_complete_type %Outer.3b5 [symbolic]
// CHECK:STDOUT:   %pattern_type.c25: type = pattern_type %Inner.6c4 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.4b9: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.600: <witness> = require_complete_type %Inner.6c4 [symbolic]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.367: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %Outer.b54: type = class_type @Outer, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.type.de5: type = generic_class_type @Inner, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.generic.598: %Inner.type.de5 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.2ce: type = class_type @Inner, @Inner(%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.263: type = pattern_type %Inner.2ce [concrete]
// CHECK:STDOUT:   %tuple.ac7: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.a6a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %.970: Core.Form = init_form %tuple.type.a6a [concrete]
// CHECK:STDOUT:   %pattern_type.ce9: type = pattern_type %tuple.type.a6a [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.335: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.3cb: <witness> = complete_type_witness %tuple.type.a6a [concrete]
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
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [concrete = constants.%Outer.generic] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_17.1: type = splice_block %.loc4_17.2 [concrete = type] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc4_17.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc13_27 (%pattern_type.c25) = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc13_27 (%pattern_type.c25) = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc13_56 (%pattern_type.eee) = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc13_56 (%pattern_type.eee) = return_slot_pattern %return.param_patt, %.loc13_56.4 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_52: type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_55: type = name_ref U, %U.loc13_17.2 [symbolic = %U.loc13_17.1 (constants.%U)]
// CHECK:STDOUT:     %.loc13_56.3: %tuple.type.24b = tuple_literal (%T.ref.loc13_52, %U.ref.loc13_55) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:     %.loc13_56.4: type = converted %.loc13_56.3, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %.loc13_56.5: Core.Form = init_form %.loc13_56.4 [symbolic = %.loc13_56.2 (constants.%.f18)]
// CHECK:STDOUT:     %.loc13_10.1: type = splice_block %.loc13_10.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc13_7: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13_10.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13_20.1: type = splice_block %.loc13_20.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.loc13_17: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %.loc13_20.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc13_17.2: type = symbolic_binding U, 1 [symbolic = %U.loc13_17.1 (constants.%U)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_45.1 (%Inner.6c4) = value_param call_param0
// CHECK:STDOUT:     %.loc13_45: type = splice_block %Inner.loc13_45.2 [symbolic = %Inner.loc13_45.1 (constants.%Inner.6c4)] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref.loc13_35: type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:       %Outer.loc13_36.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_36.1 (constants.%Outer.3b5)]
// CHECK:STDOUT:       %.loc13_37: @F.%Inner.type (%Inner.type.e77) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.92c)]
// CHECK:STDOUT:       %Inner.ref: @F.%Inner.type (%Inner.type.e77) = name_ref Inner, %.loc13_37 [symbolic = %Inner.generic (constants.%Inner.generic.92c)]
// CHECK:STDOUT:       %U.ref.loc13_44: type = name_ref U, %U.loc13_17.2 [symbolic = %U.loc13_17.1 (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_45.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_45.1 (constants.%Inner.6c4)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%Inner.loc13_45.1 (%Inner.6c4) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.a5e) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.263 = value_param_pattern [concrete]
// CHECK:STDOUT:     %p.patt: %pattern_type.263 = at_binding_pattern p, %p.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.ce9 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.ce9 = return_slot_pattern %return.param_patt, %.loc15_36.3 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_36.2: %tuple.type.24b = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35) [concrete = constants.%tuple.ac7]
// CHECK:STDOUT:     %.loc15_36.3: type = converted %.loc15_36.2, constants.%tuple.type.a6a [concrete = constants.%tuple.type.a6a]
// CHECK:STDOUT:     %.loc15_36.4: Core.Form = init_form %.loc15_36.3 [concrete = constants.%.970]
// CHECK:STDOUT:     %p.param: %Inner.2ce = value_param call_param0
// CHECK:STDOUT:     %.loc15_25: type = splice_block %Inner [concrete = constants.%Inner.2ce] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %C.ref.loc15_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Outer: type = class_type @Outer, @Outer(constants.%C) [concrete = constants.%Outer.b54]
// CHECK:STDOUT:       %.loc15_17: %Inner.type.de5 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [concrete = constants.%Inner.generic.598]
// CHECK:STDOUT:       %Inner.ref: %Inner.type.de5 = name_ref Inner, %.loc15_17 [concrete = constants.%Inner.generic.598]
// CHECK:STDOUT:       %D.ref.loc15_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [concrete = constants.%Inner.2ce]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %Inner.2ce = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.a6a = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.a6a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_14.1) [symbolic = %Inner.type (constants.%Inner.type.e77)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.e77) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.92c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.e77) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.92c)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_19.1: type = splice_block %.loc5_19.2 [concrete = type] {
// CHECK:STDOUT:         %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:         %.loc5_19.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc5_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.3b5
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: type, %U.loc5_16.2: type) {
// CHECK:STDOUT:   %U.loc5_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.6c4
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
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: type, %U.loc13_17.2: type) {
// CHECK:STDOUT:   %T.loc13_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc13_17.1: type = symbolic_binding U, 1 [symbolic = %U.loc13_17.1 (constants.%U)]
// CHECK:STDOUT:   %Outer.loc13_36.1: type = class_type @Outer, @Outer(%T.loc13_7.1) [symbolic = %Outer.loc13_36.1 (constants.%Outer.3b5)]
// CHECK:STDOUT:   %require_complete.loc13_37: <witness> = require_complete_type %Outer.loc13_36.1 [symbolic = %require_complete.loc13_37 (constants.%require_complete.f6c)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc13_7.1) [symbolic = %Inner.type (constants.%Inner.type.e77)]
// CHECK:STDOUT:   %Inner.generic: @F.%Inner.type (%Inner.type.e77) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.92c)]
// CHECK:STDOUT:   %Inner.loc13_45.1: type = class_type @Inner, @Inner(%T.loc13_7.1, %U.loc13_17.1) [symbolic = %Inner.loc13_45.1 (constants.%Inner.6c4)]
// CHECK:STDOUT:   %pattern_type.loc13_27: type = pattern_type %Inner.loc13_45.1 [symbolic = %pattern_type.loc13_27 (constants.%pattern_type.c25)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc13_7.1, %U.loc13_17.1) [symbolic = %tuple (constants.%tuple.4b9)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc13_7.1, %U.loc13_17.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc13_56.2: Core.Form = init_form %tuple.type [symbolic = %.loc13_56.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type.loc13_56: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13_56 (constants.%pattern_type.eee)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_27: <witness> = require_complete_type %Inner.loc13_45.1 [symbolic = %require_complete.loc13_27 (constants.%require_complete.600)]
// CHECK:STDOUT:   %require_complete.loc13_48: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_48 (constants.%require_complete.220)]
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2: <specific function> = specific_function constants.%F, @F(%T.loc13_7.1, %U.loc13_17.1) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.367)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%Inner.loc13_45.1 (%Inner.6c4)) -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_45.1 (%Inner.6c4) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc13_67.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.367)]
// CHECK:STDOUT:     %.loc13_56.1: ref @F.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.a5e) to %.loc13_56.1 = call %F.specific_fn.loc13_67.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %Inner.2ce) -> out %return.param: %tuple.type.a6a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.2ce = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.335]
// CHECK:STDOUT:   %.loc15_36.1: ref %tuple.type.a6a = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %tuple.type.a6a to %.loc15_36.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e77
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.92c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_36.1 => constants.%Outer.3b5
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%require_complete.f6c
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e77
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.92c
// CHECK:STDOUT:   %Inner.loc13_45.1 => constants.%Inner.6c4
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.c25
// CHECK:STDOUT:   %tuple => constants.%tuple.4b9
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc13_56.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type.loc13_56 => constants.%pattern_type.eee
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%require_complete.600
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%require_complete.220
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.367
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.de5
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.598
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%C, constants.%D) {
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%C
// CHECK:STDOUT:   %U.loc13_17.1 => constants.%D
// CHECK:STDOUT:   %Outer.loc13_36.1 => constants.%Outer.b54
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.de5
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.598
// CHECK:STDOUT:   %Inner.loc13_45.1 => constants.%Inner.2ce
// CHECK:STDOUT:   %pattern_type.loc13_27 => constants.%pattern_type.263
// CHECK:STDOUT:   %tuple => constants.%tuple.ac7
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a6a
// CHECK:STDOUT:   %.loc13_56.2 => constants.%.970
// CHECK:STDOUT:   %pattern_type.loc13_56 => constants.%pattern_type.ce9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%complete_type.3cb
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.335
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontype.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.104: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.bbe: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %WithNontype.type: type = generic_class_type @WithNontype [concrete]
// CHECK:STDOUT:   %WithNontype.generic: %WithNontype.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithNontype.f38: type = class_type @WithNontype, @WithNontype(%N.bbe) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.7b5: type = pattern_type %WithNontype.f38 [symbolic]
// CHECK:STDOUT:   %.7ea: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.837: <witness> = require_complete_type %WithNontype.f38 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.2f2: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.a64: %Int.as.Copy.impl.Op.type.2f2 = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.1a4: <witness> = impl_witness imports.%Copy.impl_witness_table.0ac, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.422: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.ce7: %Int.as.Copy.impl.Op.type.422 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.1a4) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.7e2: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.604: type = fn_type_with_self_type %Copy.WithSelf.Op.type.7e2, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.4af: <bound method> = bound_method %N.bbe, %Int.as.Copy.impl.Op.ce7 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.ce7, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f80: <bound method> = bound_method %N.bbe, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.754: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.754 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.360: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.3d6: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.0bd, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.0a4 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.360 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.3d6) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.da5: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.474: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.da5, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.82d: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.62d: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %WithNontype.87c: type = class_type @WithNontype, @WithNontype(%int_0.62d) [concrete]
// CHECK:STDOUT:   %WithNontype.val: %WithNontype.87c = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.f42: type = pattern_type %WithNontype.87c [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_0.62d) [concrete]
// CHECK:STDOUT:   %.a39: ref %WithNontype.87c = temporary invalid, %WithNontype.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %Destroy.Op.type.790b3b.2: type = fn_type @Destroy.Op.loc9_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.Op.2dd605.2: %Destroy.Op.type.790b3b.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.Op.bound: <bound method> = bound_method %.a39, %Destroy.Op.2dd605.2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.74c: <bound method> = bound_method %int_0.62d, %Int.as.Copy.impl.Op.ce7 [concrete]
// CHECK:STDOUT:   %bound_method.021: <bound method> = bound_method %int_0.62d, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .Destroy = %Core.Destroy
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.55f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.2f2) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.a64)]
// CHECK:STDOUT:   %Copy.impl_witness_table.0ac = impl_witness_table (%Core.import_ref.55f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.754 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.b42: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.048) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e2c)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.0bd = impl_witness_table (%Core.import_ref.b42), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WithNontype = %WithNontype.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WithNontype.decl: %WithNontype.type = class_decl @WithNontype [concrete = constants.%WithNontype.generic] {
// CHECK:STDOUT:     %N.patt: %pattern_type.104 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_20.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.bbe)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.104 = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.7b5) = value_param_pattern [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.7b5) = at_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32.loc6_44 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_44: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_44: Core.Form = init_form %i32.loc6_44 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %i32.loc6_10: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.bbe)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_38.1 (%WithNontype.f38) = value_param call_param0
// CHECK:STDOUT:     %.loc6_38: type = splice_block %WithNontype.loc6_38.2 [symbolic = %WithNontype.loc6_38.1 (constants.%WithNontype.f38)] {
// CHECK:STDOUT:       %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:       %N.ref.loc6_37: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.bbe)]
// CHECK:STDOUT:       %WithNontype.loc6_38.2: type = class_type @WithNontype, @WithNontype(constants.%N.bbe) [symbolic = %WithNontype.loc6_38.1 (constants.%WithNontype.f38)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_38.1 (%WithNontype.f38) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.104 = out_param_pattern [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.104 = return_slot_pattern %return.param_patt, %i32 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.7ea]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_20.2: %i32) {
// CHECK:STDOUT:   %N.loc4_20.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.bbe)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.f38
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: %i32) {
// CHECK:STDOUT:   %N.loc6_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.bbe)]
// CHECK:STDOUT:   %WithNontype.loc6_38.1: type = class_type @WithNontype, @WithNontype(%N.loc6_7.1) [symbolic = %WithNontype.loc6_38.1 (constants.%WithNontype.f38)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %WithNontype.loc6_38.1 [symbolic = %pattern_type (constants.%pattern_type.7b5)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WithNontype.loc6_38.1 [symbolic = %require_complete (constants.%require_complete.837)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.ce7 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.4af)]
// CHECK:STDOUT:   %bound_method.loc6_57.3: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_57.3 (constants.%bound_method.f80)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%WithNontype.loc6_38.1 (%WithNontype.f38)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_57: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.bbe)]
// CHECK:STDOUT:     %impl.elem0: %.604 = impl_witness_access constants.%Copy.impl_witness.1a4, element0 [concrete = constants.%Int.as.Copy.impl.Op.ce7]
// CHECK:STDOUT:     %bound_method.loc6_57.1: <bound method> = bound_method %N.ref.loc6_57, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.4af)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_57.2: <bound method> = bound_method %N.ref.loc6_57, %specific_fn [symbolic = %bound_method.loc6_57.3 (constants.%bound_method.f80)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_57.2(%N.ref.loc6_57) [symbolic = %N.loc6_7.1 (constants.%N.bbe)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> out %return.param: %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.474 = impl_witness_access constants.%ImplicitAs.impl_witness.3d6, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.ab1]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.82d]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_31.2(%int_0) [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc9_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %.loc9_31.2: %i32 = converted %int_0, %.loc9_31.1 [concrete = constants.%int_0.62d]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%int_0.62d) [concrete = constants.%WithNontype.87c]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.87c = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.87c to %.loc9_13.2 = class_init () [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %.loc9_15.1: init %WithNontype.87c = converted %.loc9_13.1, %.loc9_13.3 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_0.62d) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc9_15.2: ref %WithNontype.87c = temporary %.loc9_13.2, %.loc9_15.1 [concrete = constants.%.a39]
// CHECK:STDOUT:   %.loc9_15.3: %WithNontype.87c = acquire_value %.loc9_15.2 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc9_15.3)
// CHECK:STDOUT:   %Destroy.Op.call: init %empty_tuple.type = call constants.%Destroy.Op.bound(constants.%.a39)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.Op.loc9_15.2(%self.param: ref %WithNontype.87c) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.bbe) {
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%N.bbe
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.bbe) {
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.bbe
// CHECK:STDOUT:   %WithNontype.loc6_38.1 => constants.%WithNontype.f38
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7b5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%int_0.62d) {
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%int_0.62d
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_0.62d) {
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%int_0.62d
// CHECK:STDOUT:   %WithNontype.loc6_38.1 => constants.%WithNontype.87c
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.f42
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.74c
// CHECK:STDOUT:   %bound_method.loc6_57.3 => constants.%bound_method.021
// CHECK:STDOUT: }
// CHECK:STDOUT:
