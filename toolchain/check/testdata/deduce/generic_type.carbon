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

class C(T: type) {}
class D {}

fn F[T: type](p: C(T)) -> T { return F(p); }

fn G(p: C(D)) -> D {
  return F(p);
}

// --- interface.carbon

library "[[@TEST_NAME]]";

class I(T: type) {}
class C {}

fn F[T: type](p: I(T)) -> C { return F(p); }

fn G(p: I(C)) -> C {
  return F(p);
}

// --- nested.carbon

library "[[@TEST_NAME]]";

class Outer(T: type) {
  class Inner(U: type) {}
}

class C {}
class D {}

// C++ doesn't permit deducing `T` here because `Outer` might be specialized.
// But that's not possible in Carbon, so we can deduce `T`.
fn F[T: type, U: type](p: Outer(T).Inner(U)) -> (T, U) { return F(p); }

fn G(p: Outer(C).Inner(D)) -> (C, D) {
  return F(p);
}

// --- nontype.carbon

library "[[@TEST_NAME]]";

class WithNontype(N: i32) {}

fn F[N: i32](unused x: WithNontype(N)) -> i32 { return N; }

fn G() -> i32 {
  return F({} as WithNontype(0));
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = parameterized_class_name_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.7e5: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.764: type = pattern_type %C.7e5 [symbolic]
// CHECK:STDOUT:   %p.param_patt.12a: %pattern_type.764 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.8a3: %pattern_type.764 = wrapper_binding_pattern p, %p.param_patt.12a [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.466: <witness> = require_complete_type %C.7e5 [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.eab: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %C.278: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %pattern_type.487: type = pattern_type %C.278 [concrete]
// CHECK:STDOUT:   %p.param_patt.083: %pattern_type.487 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.e8d: %pattern_type.487 = wrapper_binding_pattern p, %p.param_patt.083 [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.eae: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e38: %pattern_type.d8d = return_slot_pattern %return.param_patt.eae, %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.8e7: <specific function> = specific_function %F, @F(%D) [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_10.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: type = splice_block %.loc4_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.param_patt.loc7_16.1: @F.%pattern_type.loc7_16 (%pattern_type.764) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.12a)]
// CHECK:STDOUT:     %p.patt.loc7_16.1: @F.%pattern_type.loc7_16 (%pattern_type.764) = wrapper_binding_pattern p, %p.param_patt.loc7_16.1 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.8a3)]
// CHECK:STDOUT:     %return.param_patt.loc7_27.1: @F.%pattern_type.loc7_27 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc7_27.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:     %return.patt.loc7_24.1: @F.%pattern_type.loc7_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_27.1, %T.ref.loc7_27 [symbolic = %return.patt.loc7_24.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_27: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7_27.3: Core.Form = init_form %T.ref.loc7_27 [symbolic = %.loc7_27.2 (constants.%.184)]
// CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_21.1 (%C.7e5) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21: type = splice_block %C.loc7_21.2 [symbolic = %C.loc7_21.1 (constants.%C.7e5)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc7_20: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_21.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_21.1 (constants.%C.7e5)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%C.loc7_21.1 (%C.7e5) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.487 = value_param_pattern [concrete = constants.%p.param_patt.083]
// CHECK:STDOUT:     %p.patt: %pattern_type.487 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.e8d]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc9_18 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %D.ref.loc9_18 [concrete = constants.%.592]
// CHECK:STDOUT:     %p.param: %C.278 = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %C [concrete = constants.%C.278] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc9_11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.278]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %C.278 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_10.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.7e5
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
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc7_21.1: type = class_type @C, @C(%T.loc7_7.1) [symbolic = %C.loc7_21.1 (constants.%C.7e5)]
// CHECK:STDOUT:   %pattern_type.loc7_16: type = pattern_type %C.loc7_21.1 [symbolic = %pattern_type.loc7_16 (constants.%pattern_type.764)]
// CHECK:STDOUT:   %p.param_patt.loc7_16.2: @F.%pattern_type.loc7_16 (%pattern_type.764) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.12a)]
// CHECK:STDOUT:   %p.patt.loc7_16.2: @F.%pattern_type.loc7_16 (%pattern_type.764) = wrapper_binding_pattern p, %p.param_patt.loc7_16.2 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.8a3)]
// CHECK:STDOUT:   %.loc7_27.2: Core.Form = init_form %T.loc7_7.1 [symbolic = %.loc7_27.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_27: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc7_27.2: @F.%pattern_type.loc7_27 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc7_27.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc7_24.2: @F.%pattern_type.loc7_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_27.2, %T.loc7_7.1 [symbolic = %return.patt.loc7_24.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16: <witness> = require_complete_type %C.loc7_21.1 [symbolic = %require_complete.loc7_16 (constants.%require_complete.466)]
// CHECK:STDOUT:   %require_complete.loc7_24: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7_24 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.eab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%C.loc7_21.1 (%C.7e5)) -> out %return.param: @F.%T.loc7_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_21.1 (%C.7e5) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_38.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.eab)]
// CHECK:STDOUT:     %.loc7_27.1: ref @F.%T.loc7_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_7.1 (%T) to %.loc7_27.1 = call %F.specific_fn.loc7_38.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %C.278) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.278 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn.8e7]
// CHECK:STDOUT:   %.loc9_18.1: ref %D = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %D to %.loc9_18.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %C.loc7_21.1 => constants.%C.7e5
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.764
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.12a
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.8a3
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc7_27.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc7_24.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16 => constants.%require_complete.466
// CHECK:STDOUT:   %require_complete.loc7_24 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.eab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%D
// CHECK:STDOUT:   %C.loc7_21.1 => constants.%C.278
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.487
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.083
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.e8d
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.592
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %return.param_patt.loc7_27.2 => constants.%return.param_patt.eae
// CHECK:STDOUT:   %return.patt.loc7_24.2 => constants.%return.patt.e38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc7_24 => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.8e7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = parameterized_class_name_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.d7e: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.469: type = pattern_type %I.d7e [symbolic]
// CHECK:STDOUT:   %p.param_patt.fb3: %pattern_type.469 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.eb6: %pattern_type.469 = wrapper_binding_pattern p, %p.param_patt.fb3 [symbolic]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.d7e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.eab: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %I.46d: type = class_type @I, @I(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.347: type = pattern_type %I.46d [concrete]
// CHECK:STDOUT:   %p.param_patt.1e6: %pattern_type.347 = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.6a0: %pattern_type.347 = wrapper_binding_pattern p, %p.param_patt.1e6 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.e0d: <specific function> = specific_function %F, @F(%C) [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_10.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_12.1: type = splice_block %.loc4_12.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_12.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_10.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.param_patt.loc7_16.1: @F.%pattern_type (%pattern_type.469) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.fb3)]
// CHECK:STDOUT:     %p.patt.loc7_16.1: @F.%pattern_type (%pattern_type.469) = wrapper_binding_pattern p, %p.param_patt.loc7_16.1 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.eb6)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc7_27.2: Core.Form = init_form %C.ref [concrete = constants.%.a44]
// CHECK:STDOUT:     %.loc7_9.1: type = splice_block %.loc7_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc7_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc7_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_21.1 (%I.d7e) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21: type = splice_block %I.loc7_21.2 [symbolic = %I.loc7_21.1 (constants.%I.d7e)] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %I.loc7_21.2: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_21.1 (constants.%I.d7e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%I.loc7_21.1 (%I.d7e) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.347 = value_param_pattern [concrete = constants.%p.param_patt.1e6]
// CHECK:STDOUT:     %p.patt: %pattern_type.347 = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.6a0]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc9_18 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %C.ref.loc9_18 [concrete = constants.%.a44]
// CHECK:STDOUT:     %p.param: %I.46d = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I [concrete = constants.%I.46d] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %C.ref.loc9_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %I: type = class_type @I, @I(constants.%C) [concrete = constants.%I.46d]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %I.46d = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_10.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_10.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_10.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_10.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_10.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.d7e
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
// CHECK:STDOUT:   %T.patt.loc7_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc7_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:   %I.loc7_21.1: type = class_type @I, @I(%T.loc7_7.1) [symbolic = %I.loc7_21.1 (constants.%I.d7e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.loc7_21.1 [symbolic = %pattern_type (constants.%pattern_type.469)]
// CHECK:STDOUT:   %p.param_patt.loc7_16.2: @F.%pattern_type (%pattern_type.469) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.fb3)]
// CHECK:STDOUT:   %p.patt.loc7_16.2: @F.%pattern_type (%pattern_type.469) = wrapper_binding_pattern p, %p.param_patt.loc7_16.2 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.eb6)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.loc7_21.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.eab)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%I.loc7_21.1 (%I.d7e)) -> out %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_21.1 (%I.d7e) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_38.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.eab)]
// CHECK:STDOUT:     %.loc7_27.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init %C to %.loc7_27.1 = call %F.specific_fn.loc7_38.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %I.46d) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.46d = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.e0d]
// CHECK:STDOUT:   %.loc9_18.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %C to %.loc9_18.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%T
// CHECK:STDOUT:   %I.loc7_21.1 => constants.%I.d7e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.469
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.fb3
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.eb6
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.eab
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_10.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_10.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc7_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc7_7.1 => constants.%C
// CHECK:STDOUT:   %I.loc7_21.1 => constants.%I.46d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.347
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.1e6
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.6a0
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.e0d
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %pattern_type.9a5: type = pattern_type type [concrete]
// CHECK:STDOUT:   %T.patt: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = parameterized_class_name_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.899: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.884: type = parameterized_class_name_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.9ef: %Inner.type.884 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.7d9: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %require_complete.d76: <witness> = require_complete_type %Outer.899 [symbolic]
// CHECK:STDOUT:   %pattern_type.195: type = pattern_type %Inner.7d9 [symbolic]
// CHECK:STDOUT:   %p.param_patt.aeb: %pattern_type.195 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.ecc: %pattern_type.195 = wrapper_binding_pattern p, %p.param_patt.aeb [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.fa2: %tuple.type.693 = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %return.param_patt.a41: %pattern_type.eee = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.453: %pattern_type.eee = return_slot_pattern %return.param_patt.a41, %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.168: <witness> = require_complete_type %Inner.7d9 [symbolic]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.19a: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %Outer.70e: type = class_type @Outer, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.type.95e: type = parameterized_class_name_type @Inner, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.generic.fe1: %Inner.type.95e = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.4e0: type = class_type @Inner, @Inner(%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.f8b: type = pattern_type %Inner.4e0 [concrete]
// CHECK:STDOUT:   %p.param_patt.c5b: %pattern_type.f8b = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.74e: %pattern_type.f8b = wrapper_binding_pattern p, %p.param_patt.c5b [concrete]
// CHECK:STDOUT:   %tuple.919: %tuple.type.693 = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.af1: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %.71f: Core.Form = init_form %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %return.param_patt.fc8: %pattern_type.c66 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.964: %pattern_type.c66 = return_slot_pattern %return.param_patt.fc8, %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.fca: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.6db: <witness> = complete_type_witness %tuple.type.af1 [concrete]
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
// CHECK:STDOUT:     %T.patt.loc4_14.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_16.1: type = splice_block %.loc4_16.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc4_16.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc4_14.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_7.1: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_7.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc13_16.1: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %p.param_patt.loc13_25.1: @F.%pattern_type.loc13_25 (%pattern_type.195) = value_param_pattern [symbolic = %p.param_patt.loc13_25.2 (constants.%p.param_patt.aeb)]
// CHECK:STDOUT:     %p.patt.loc13_25.1: @F.%pattern_type.loc13_25 (%pattern_type.195) = wrapper_binding_pattern p, %p.param_patt.loc13_25.1 [symbolic = %p.patt.loc13_25.2 (constants.%p.patt.ecc)]
// CHECK:STDOUT:     %return.param_patt.loc13_54.1: @F.%pattern_type.loc13_54 (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc13_54.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:     %return.patt.loc13_46.1: @F.%pattern_type.loc13_54 (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc13_54.1, %.loc13_54.4 [symbolic = %return.patt.loc13_46.2 (constants.%return.patt.453)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_50: type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_53: type = name_ref U, %U.loc13_16.2 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:     %.loc13_54.3: %tuple.type.693 = tuple_literal (%T.ref.loc13_50, %U.ref.loc13_53) [symbolic = %tuple (constants.%tuple.fa2)]
// CHECK:STDOUT:     %.loc13_54.4: type = converted %.loc13_54.3, constants.%tuple.type.a5e [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:     %.loc13_54.5: Core.Form = init_form %.loc13_54.4 [symbolic = %.loc13_54.2 (constants.%.f18)]
// CHECK:STDOUT:     %.loc13_9.1: type = splice_block %.loc13_9.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc13_7: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc13_9.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc13_7.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:     %.loc13_18.1: type = splice_block %.loc13_18.2 [concrete = type] {
// CHECK:STDOUT:       %.Self.frozen.loc13_16: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %.loc13_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %U.loc13_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_43.1 (%Inner.7d9) = value_param call_param0
// CHECK:STDOUT:     %.loc13_43: type = splice_block %Inner.loc13_43.2 [symbolic = %Inner.loc13_43.1 (constants.%Inner.7d9)] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref.loc13_33: type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:       %Outer.loc13_34.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_34.1 (constants.%Outer.899)]
// CHECK:STDOUT:       %.loc13_35: @F.%Inner.type (%Inner.type.884) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.9ef)]
// CHECK:STDOUT:       %Inner.ref: @F.%Inner.type (%Inner.type.884) = name_ref Inner, %.loc13_35 [symbolic = %Inner.generic (constants.%Inner.generic.9ef)]
// CHECK:STDOUT:       %U.ref.loc13_42: type = name_ref U, %U.loc13_16.2 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_43.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_43.1 (constants.%Inner.7d9)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%Inner.loc13_43.1 (%Inner.7d9) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.a5e) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.f8b = value_param_pattern [concrete = constants.%p.param_patt.c5b]
// CHECK:STDOUT:     %p.patt: %pattern_type.f8b = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.74e]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c66 = out_param_pattern [concrete = constants.%return.param_patt.fc8]
// CHECK:STDOUT:     %return.patt: %pattern_type.c66 = return_slot_pattern %return.param_patt, %.loc15_36.3 [concrete = constants.%return.patt.964]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_36.2: %tuple.type.693 = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35) [concrete = constants.%tuple.919]
// CHECK:STDOUT:     %.loc15_36.3: type = converted %.loc15_36.2, constants.%tuple.type.af1 [concrete = constants.%tuple.type.af1]
// CHECK:STDOUT:     %.loc15_36.4: Core.Form = init_form %.loc15_36.3 [concrete = constants.%.71f]
// CHECK:STDOUT:     %p.param: %Inner.4e0 = value_param call_param0
// CHECK:STDOUT:     %.loc15_25: type = splice_block %Inner [concrete = constants.%Inner.4e0] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %C.ref.loc15_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Outer: type = class_type @Outer, @Outer(constants.%C) [concrete = constants.%Outer.70e]
// CHECK:STDOUT:       %.loc15_17: %Inner.type.95e = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [concrete = constants.%Inner.generic.fe1]
// CHECK:STDOUT:       %Inner.ref: %Inner.type.95e = name_ref Inner, %.loc15_17 [concrete = constants.%Inner.generic.fe1]
// CHECK:STDOUT:       %D.ref.loc15_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [concrete = constants.%Inner.4e0]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %Inner.4e0 = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.af1 = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.af1 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_14.2: type) {
// CHECK:STDOUT:   %T.patt.loc4_14.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_14.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc4_14.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_14.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T.loc4_14.1) [symbolic = %Inner.type (constants.%Inner.type.884)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.884) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.9ef)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.884) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.9ef)] {
// CHECK:STDOUT:       %U.patt.loc5_16.1: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.loc5_18.1: type = splice_block %.loc5_18.2 [concrete = type] {
// CHECK:STDOUT:         %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:         %.loc5_18.2: type = type_literal type [concrete = type]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %U.loc5_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.899
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_14.2: type, %U.loc5_16.2: type) {
// CHECK:STDOUT:   %U.patt.loc5_16.2: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc5_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_16.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.7d9
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
// CHECK:STDOUT: generic fn @F(%T.loc13_7.2: type, %U.loc13_16.2: type) {
// CHECK:STDOUT:   %T.patt.loc13_7.2: %pattern_type.9a5 = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_7.2 (constants.%T.patt)]
// CHECK:STDOUT:   %T.loc13_7.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:   %U.patt.loc13_16.2: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %U.loc13_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:   %Outer.loc13_34.1: type = class_type @Outer, @Outer(%T.loc13_7.1) [symbolic = %Outer.loc13_34.1 (constants.%Outer.899)]
// CHECK:STDOUT:   %require_complete.loc13_35: <witness> = require_complete_type %Outer.loc13_34.1 [symbolic = %require_complete.loc13_35 (constants.%require_complete.d76)]
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T.loc13_7.1) [symbolic = %Inner.type (constants.%Inner.type.884)]
// CHECK:STDOUT:   %Inner.generic: @F.%Inner.type (%Inner.type.884) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.9ef)]
// CHECK:STDOUT:   %Inner.loc13_43.1: type = class_type @Inner, @Inner(%T.loc13_7.1, %U.loc13_16.1) [symbolic = %Inner.loc13_43.1 (constants.%Inner.7d9)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %Inner.loc13_43.1 [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.195)]
// CHECK:STDOUT:   %p.param_patt.loc13_25.2: @F.%pattern_type.loc13_25 (%pattern_type.195) = value_param_pattern [symbolic = %p.param_patt.loc13_25.2 (constants.%p.param_patt.aeb)]
// CHECK:STDOUT:   %p.patt.loc13_25.2: @F.%pattern_type.loc13_25 (%pattern_type.195) = wrapper_binding_pattern p, %p.param_patt.loc13_25.2 [symbolic = %p.patt.loc13_25.2 (constants.%p.patt.ecc)]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%T.loc13_7.1, %U.loc13_16.1) [symbolic = %tuple (constants.%tuple.fa2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc13_7.1, %U.loc13_16.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc13_54.2: Core.Form = init_form %tuple.type [symbolic = %.loc13_54.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type.loc13_54: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13_54 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc13_54.2: @F.%pattern_type.loc13_54 (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc13_54.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:   %return.patt.loc13_46.2: @F.%pattern_type.loc13_54 (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc13_54.2, %tuple.type [symbolic = %return.patt.loc13_46.2 (constants.%return.patt.453)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_25: <witness> = require_complete_type %Inner.loc13_43.1 [symbolic = %require_complete.loc13_25 (constants.%require_complete.168)]
// CHECK:STDOUT:   %require_complete.loc13_46: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_46 (constants.%require_complete.220)]
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2: <specific function> = specific_function constants.%F, @F(%T.loc13_7.1, %U.loc13_16.1) [symbolic = %F.specific_fn.loc13_65.2 (constants.%F.specific_fn.19a)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%Inner.loc13_43.1 (%Inner.7d9)) -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_43.1 (%Inner.7d9) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc13_65.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc13_65.2 (constants.%F.specific_fn.19a)]
// CHECK:STDOUT:     %.loc13_54.1: ref @F.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.a5e) to %.loc13_54.1 = call %F.specific_fn.loc13_65.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %Inner.4e0) -> out %return.param: %tuple.type.af1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.4e0 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.fca]
// CHECK:STDOUT:   %.loc15_36.1: ref %tuple.type.af1 = splice_block %return.param {}
// CHECK:STDOUT:   %F.call: init %tuple.type.af1 to %.loc15_36.1 = call %F.specific_fn(%p.ref)
// CHECK:STDOUT:   return %F.call to %return.param
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.884
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.9ef
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.patt.loc5_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.patt.loc13_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%T
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_16.1 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_34.1 => constants.%Outer.899
// CHECK:STDOUT:   %require_complete.loc13_35 => constants.%require_complete.d76
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.884
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.9ef
// CHECK:STDOUT:   %Inner.loc13_43.1 => constants.%Inner.7d9
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.195
// CHECK:STDOUT:   %p.param_patt.loc13_25.2 => constants.%p.param_patt.aeb
// CHECK:STDOUT:   %p.patt.loc13_25.2 => constants.%p.patt.ecc
// CHECK:STDOUT:   %tuple => constants.%tuple.fa2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc13_54.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type.loc13_54 => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc13_54.2 => constants.%return.param_patt.a41
// CHECK:STDOUT:   %return.patt.loc13_46.2 => constants.%return.patt.453
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_25 => constants.%require_complete.168
// CHECK:STDOUT:   %require_complete.loc13_46 => constants.%require_complete.220
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2 => constants.%F.specific_fn.19a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.95e
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.fe1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%C, constants.%D) {
// CHECK:STDOUT:   %U.patt.loc5_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc5_16.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.patt.loc13_7.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc13_7.1 => constants.%C
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%U.patt
// CHECK:STDOUT:   %U.loc13_16.1 => constants.%D
// CHECK:STDOUT:   %Outer.loc13_34.1 => constants.%Outer.70e
// CHECK:STDOUT:   %require_complete.loc13_35 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.95e
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.fe1
// CHECK:STDOUT:   %Inner.loc13_43.1 => constants.%Inner.4e0
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.f8b
// CHECK:STDOUT:   %p.param_patt.loc13_25.2 => constants.%p.param_patt.c5b
// CHECK:STDOUT:   %p.patt.loc13_25.2 => constants.%p.patt.74e
// CHECK:STDOUT:   %tuple => constants.%tuple.919
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.af1
// CHECK:STDOUT:   %.loc13_54.2 => constants.%.71f
// CHECK:STDOUT:   %pattern_type.loc13_54 => constants.%pattern_type.c66
// CHECK:STDOUT:   %return.param_patt.loc13_54.2 => constants.%return.param_patt.fc8
// CHECK:STDOUT:   %return.patt.loc13_46.2 => constants.%return.patt.964
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_25 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_46 => constants.%complete_type.6db
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2 => constants.%F.specific_fn.fca
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontype.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self.frozen: type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = parameterized_class_name_type @Int [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %N.fe9: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %pattern_type.df9: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.282: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.06f: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %WithNontype.type: type = parameterized_class_name_type @WithNontype [concrete]
// CHECK:STDOUT:   %WithNontype.generic: %WithNontype.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithNontype.a4f: type = class_type @WithNontype, @WithNontype(%N.06f) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.4fb: type = pattern_type %WithNontype.a4f [symbolic]
// CHECK:STDOUT:   %x.param_patt.0ad: %pattern_type.4fb = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.009: %pattern_type.4fb = wrapper_binding_pattern x, %x.param_patt.0ad [symbolic]
// CHECK:STDOUT:   %.4e1: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.973: %pattern_type.df9 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.a21: %pattern_type.df9 = return_slot_pattern %return.param_patt.973, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.bd9: <witness> = require_complete_type %WithNontype.a4f [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b5e: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.f38: %Int.as.Copy.impl.Op.type.b5e = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.4c3: <witness> = impl_witness imports.%Copy.impl_witness_table.c8d, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.b4b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.175: %Int.as.Copy.impl.Op.type.b4b = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.4c3) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.67d: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.187: type = fn_type_with_self_type %Copy.WithSelf.Op.type.67d, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.f5e: <bound method> = bound_method %N.06f, %Int.as.Copy.impl.Op.175 [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.175, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.450: <bound method> = bound_method %N.06f, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cb7: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.e87: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.1af, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.186: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3e5 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.cb7 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.e87) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.ac8: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.208: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.ac8, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.186 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.186, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.be3: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.1b1: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %WithNontype.2d9: type = class_type @WithNontype, @WithNontype(%int_0.1b1) [concrete]
// CHECK:STDOUT:   %WithNontype.val: %WithNontype.2d9 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.31b: type = pattern_type %WithNontype.2d9 [concrete]
// CHECK:STDOUT:   %x.param_patt.5d4: %pattern_type.31b = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.f06: %pattern_type.31b = wrapper_binding_pattern x, %x.param_patt.5d4 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_0.1b1) [concrete]
// CHECK:STDOUT:   %.38d: ref %WithNontype.2d9 = temporary invalid, %WithNontype.val [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %self.param_patt.52f: %pattern_type.a96 = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.4b1: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt.52f [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_15.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.1: type = fn_type @Destroy.WithSelf.Op.loc9_15.1 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.1: %Destroy.WithSelf.Op.type.ef016f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %self.param_patt.814: %pattern_type.31b = ref_param_pattern [concrete]
// CHECK:STDOUT:   %self.patt.c00: %pattern_type.31b = wrapper_binding_pattern self, %self.param_patt.814 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2: type = fn_type @Destroy.WithSelf.SubobjectDestroy.loc9_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.d01daf.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.type.ef016f.2: type = fn_type @Destroy.WithSelf.Op.loc9_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.Op.403171.2: %Destroy.WithSelf.Op.type.ef016f.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.type.fbceb5.2: type = fn_type @Destroy.WithSelf.SelfDestruct.loc9_15.2 [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.db3fdb.2: %Destroy.WithSelf.SelfDestruct.type.fbceb5.2 = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.bound: <bound method> = bound_method %.38d, %Destroy.WithSelf.SelfDestruct.db3fdb.2 [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.019: <bound method> = bound_method %int_0.1b1, %Int.as.Copy.impl.Op.175 [concrete]
// CHECK:STDOUT:   %bound_method.8cd: <bound method> = bound_method %int_0.1b1, %Int.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.7f2: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.b5e) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.f38)]
// CHECK:STDOUT:   %Copy.impl_witness_table.c8d = impl_witness_table (%Core.import_ref.7f2), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.7e0: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.c7c) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f5e)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.1af = impl_witness_table (%Core.import_ref.7e0), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.Destroy: type = import_ref Core//prelude/parts/destroy, Destroy, loaded [concrete = constants.%Destroy.type]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generated {
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.1: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.1 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_15.1 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.1: %Destroy.WithSelf.Op.type.ef016f.1 = fn_decl @Destroy.WithSelf.Op.loc9_15.1 [concrete = constants.%Destroy.WithSelf.Op.403171.1] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.a96 = ref_param_pattern [concrete = constants.%self.param_patt.52f]
// CHECK:STDOUT:     %self.patt: %pattern_type.a96 = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.4b1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %empty_struct_type = ref_param call_param0
// CHECK:STDOUT:     %self: ref %empty_struct_type = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.decl.763c71.2: %Destroy.WithSelf.SubobjectDestroy.type.dfcbdb.2 = fn_decl @Destroy.WithSelf.SubobjectDestroy.loc9_15.2 [concrete = constants.%Destroy.WithSelf.SubobjectDestroy.d01daf.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.31b = ref_param_pattern [concrete = constants.%self.param_patt.814]
// CHECK:STDOUT:     %self.patt: %pattern_type.31b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c00]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %WithNontype.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %WithNontype.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Destroy.WithSelf.Op.decl.5f94cf.2: %Destroy.WithSelf.Op.type.ef016f.2 = fn_decl @Destroy.WithSelf.Op.loc9_15.2 [concrete = constants.%Destroy.WithSelf.Op.403171.2] {
// CHECK:STDOUT:     %self.param_patt: %pattern_type.31b = ref_param_pattern [concrete = constants.%self.param_patt.814]
// CHECK:STDOUT:     %self.patt: %pattern_type.31b = wrapper_binding_pattern self, %self.param_patt [concrete = constants.%self.patt.c00]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %self.param: ref %WithNontype.2d9 = ref_param call_param0
// CHECK:STDOUT:     %self: ref %WithNontype.2d9 = wrapper_binding self, %self.param
// CHECK:STDOUT:   }
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
// CHECK:STDOUT:     %N.patt.loc4_20.1: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_20.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.06f)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc6_7.1: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.282)]
// CHECK:STDOUT:     %x.param_patt.loc6_22.1: @F.%pattern_type (%pattern_type.4fb) = value_param_pattern [symbolic = %x.param_patt.loc6_22.2 (constants.%x.param_patt.0ad)]
// CHECK:STDOUT:     %x.patt.loc6_22.1: @F.%pattern_type (%pattern_type.4fb) = wrapper_binding_pattern x, %x.param_patt.loc6_22.1 [symbolic = %x.patt.loc6_22.2 (constants.%x.patt.009)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete = constants.%return.param_patt.973]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %i32.loc6_43 [concrete = constants.%return.patt.a21]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_43: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_43: Core.Form = init_form %i32.loc6_43 [concrete = constants.%.4e1]
// CHECK:STDOUT:     %.loc6_9: type = splice_block %i32.loc6_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32.loc6_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.06f)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_37.1 (%WithNontype.a4f) = value_param call_param0
// CHECK:STDOUT:     %.loc6_37: type = splice_block %WithNontype.loc6_37.2 [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.a4f)] {
// CHECK:STDOUT:       %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:       %N.ref.loc6_36: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.06f)]
// CHECK:STDOUT:       %WithNontype.loc6_37.2: type = class_type @WithNontype, @WithNontype(constants.%N.06f) [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.a4f)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_37.1 (%WithNontype.a4f) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.df9 = out_param_pattern [concrete = constants.%return.param_patt.973]
// CHECK:STDOUT:     %return.patt: %pattern_type.df9 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.a21]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.4e1]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_20.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc4_20.2: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   %N.loc4_20.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.06f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.a4f
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc6_7.2: %pattern_type.df9 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.282)]
// CHECK:STDOUT:   %N.loc6_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.06f)]
// CHECK:STDOUT:   %WithNontype.loc6_37.1: type = class_type @WithNontype, @WithNontype(%N.loc6_7.1) [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.a4f)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %WithNontype.loc6_37.1 [symbolic = %pattern_type (constants.%pattern_type.4fb)]
// CHECK:STDOUT:   %x.param_patt.loc6_22.2: @F.%pattern_type (%pattern_type.4fb) = value_param_pattern [symbolic = %x.param_patt.loc6_22.2 (constants.%x.param_patt.0ad)]
// CHECK:STDOUT:   %x.patt.loc6_22.2: @F.%pattern_type (%pattern_type.4fb) = wrapper_binding_pattern x, %x.param_patt.loc6_22.2 [symbolic = %x.patt.loc6_22.2 (constants.%x.patt.009)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WithNontype.loc6_37.1 [symbolic = %require_complete (constants.%require_complete.bd9)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.175 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.f5e)]
// CHECK:STDOUT:   %bound_method.loc6_56.3: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_56.3 (constants.%bound_method.450)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%WithNontype.loc6_37.1 (%WithNontype.a4f)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_56: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.06f)]
// CHECK:STDOUT:     %impl.elem0: %.187 = impl_witness_access constants.%Copy.impl_witness.4c3, element0 [concrete = constants.%Int.as.Copy.impl.Op.175]
// CHECK:STDOUT:     %bound_method.loc6_56.1: <bound method> = bound_method %N.ref.loc6_56, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.f5e)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_56.2: <bound method> = bound_method %N.ref.loc6_56, %specific_fn [symbolic = %bound_method.loc6_56.3 (constants.%bound_method.450)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_56.2(%N.ref.loc6_56) [symbolic = %N.loc6_7.1 (constants.%N.06f)]
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
// CHECK:STDOUT:   %impl.elem0: %.208 = impl_witness_access constants.%ImplicitAs.impl_witness.e87, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.186]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.be3]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_31.2(%int_0) [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc9_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %.loc9_31.2: %i32 = converted %int_0, %.loc9_31.1 [concrete = constants.%int_0.1b1]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%int_0.1b1) [concrete = constants.%WithNontype.2d9]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.2d9 = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.2d9 to %.loc9_13.2 = class_init () [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %.loc9_15.1: init %WithNontype.2d9 = converted %.loc9_13.1, %.loc9_13.3 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_0.1b1) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc9_15.2: ref %WithNontype.2d9 = temporary %.loc9_13.2, %.loc9_15.1 [concrete = constants.%.38d]
// CHECK:STDOUT:   %.loc9_15.3: %WithNontype.2d9 = acquire_value %.loc9_15.2 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc9_15.3)
// CHECK:STDOUT:   %Destroy.WithSelf.SelfDestruct.call: init %empty_tuple.type = call constants.%Destroy.WithSelf.SelfDestruct.bound(constants.%.38d)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_15.1(%self.param: ref %empty_struct_type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.1(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.1(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SubobjectDestroy.loc9_15.2(%self.param: ref %WithNontype.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.Op.loc9_15.2(%self.param: ref %WithNontype.2d9) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @Destroy.WithSelf.SelfDestruct.loc9_15.2(%self.param: ref %WithNontype.2d9) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Destroy.WithSelf.Op.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.Op.decl.5f94cf.2(%self.param)
// CHECK:STDOUT:   %Destroy.WithSelf.SubobjectDestroy.call: init %empty_tuple.type = call generated.%Destroy.WithSelf.SubobjectDestroy.decl.763c71.2(%self.param)
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.06f) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%N.06f
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.06f) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.06f
// CHECK:STDOUT:   %WithNontype.loc6_37.1 => constants.%WithNontype.a4f
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.4fb
// CHECK:STDOUT:   %x.param_patt.loc6_22.2 => constants.%x.param_patt.0ad
// CHECK:STDOUT:   %x.patt.loc6_22.2 => constants.%x.patt.009
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%int_0.1b1) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%int_0.1b1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_0.1b1) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.282
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%int_0.1b1
// CHECK:STDOUT:   %WithNontype.loc6_37.1 => constants.%WithNontype.2d9
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.31b
// CHECK:STDOUT:   %x.param_patt.loc6_22.2 => constants.%x.param_patt.5d4
// CHECK:STDOUT:   %x.patt.loc6_22.2 => constants.%x.patt.f06
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.019
// CHECK:STDOUT:   %bound_method.loc6_56.3 => constants.%bound_method.8cd
// CHECK:STDOUT: }
// CHECK:STDOUT:
