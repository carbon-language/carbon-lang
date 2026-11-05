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
// CHECK:STDOUT:   %C.04a: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.563: type = pattern_type %C.04a [symbolic]
// CHECK:STDOUT:   %p.param_patt.6eb: %pattern_type.563 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.41e: %pattern_type.563 = wrapper_binding_pattern p, %p.param_patt.6eb [symbolic]
// CHECK:STDOUT:   %.184: Core.Form = init_form %T [symbolic]
// CHECK:STDOUT:   %pattern_type.51d: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %return.param_patt.41d: %pattern_type.51d = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.b39: %pattern_type.51d = return_slot_pattern %return.param_patt.41d, %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.095: <witness> = require_complete_type %C.04a [symbolic]
// CHECK:STDOUT:   %require_complete.944: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %F.specific_fn.c05: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %C.56f: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %pattern_type.e3f: type = pattern_type %C.56f [concrete]
// CHECK:STDOUT:   %p.param_patt.0b5: %pattern_type.e3f = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.8c3: %pattern_type.e3f = wrapper_binding_pattern p, %p.param_patt.0b5 [concrete]
// CHECK:STDOUT:   %.592: Core.Form = init_form %D [concrete]
// CHECK:STDOUT:   %pattern_type.d8d: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %return.param_patt.eae: %pattern_type.d8d = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.e38: %pattern_type.d8d = return_slot_pattern %return.param_patt.eae, %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.8e2: <specific function> = specific_function %F, @F(%D) [concrete]
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
// CHECK:STDOUT:     %p.param_patt.loc7_16.1: @F.%pattern_type.loc7_16 (%pattern_type.563) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.6eb)]
// CHECK:STDOUT:     %p.patt.loc7_16.1: @F.%pattern_type.loc7_16 (%pattern_type.563) = wrapper_binding_pattern p, %p.param_patt.loc7_16.1 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.41e)]
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
// CHECK:STDOUT:     %p.param: @F.%C.loc7_21.1 (%C.04a) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21: type = splice_block %C.loc7_21.2 [symbolic = %C.loc7_21.1 (constants.%C.04a)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc7_20: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_21.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_21.1 (constants.%C.04a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%C.loc7_21.1 (%C.04a) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_7.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_7.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.e3f = value_param_pattern [concrete = constants.%p.param_patt.0b5]
// CHECK:STDOUT:     %p.patt: %pattern_type.e3f = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.8c3]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d8d = out_param_pattern [concrete = constants.%return.param_patt.eae]
// CHECK:STDOUT:     %return.patt: %pattern_type.d8d = return_slot_pattern %return.param_patt, %D.ref.loc9_18 [concrete = constants.%return.patt.e38]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %D.ref.loc9_18 [concrete = constants.%.592]
// CHECK:STDOUT:     %p.param: %C.56f = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %C [concrete = constants.%C.56f] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc9_11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.56f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %C.56f = wrapper_binding p, %p.param
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
// CHECK:STDOUT:     .Self = constants.%C.04a
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
// CHECK:STDOUT:   %C.loc7_21.1: type = class_type @C, @C(%T.loc7_7.1) [symbolic = %C.loc7_21.1 (constants.%C.04a)]
// CHECK:STDOUT:   %pattern_type.loc7_16: type = pattern_type %C.loc7_21.1 [symbolic = %pattern_type.loc7_16 (constants.%pattern_type.563)]
// CHECK:STDOUT:   %p.param_patt.loc7_16.2: @F.%pattern_type.loc7_16 (%pattern_type.563) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.6eb)]
// CHECK:STDOUT:   %p.patt.loc7_16.2: @F.%pattern_type.loc7_16 (%pattern_type.563) = wrapper_binding_pattern p, %p.param_patt.loc7_16.2 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.41e)]
// CHECK:STDOUT:   %.loc7_27.2: Core.Form = init_form %T.loc7_7.1 [symbolic = %.loc7_27.2 (constants.%.184)]
// CHECK:STDOUT:   %pattern_type.loc7_27: type = pattern_type %T.loc7_7.1 [symbolic = %pattern_type.loc7_27 (constants.%pattern_type.51d)]
// CHECK:STDOUT:   %return.param_patt.loc7_27.2: @F.%pattern_type.loc7_27 (%pattern_type.51d) = out_param_pattern [symbolic = %return.param_patt.loc7_27.2 (constants.%return.param_patt.41d)]
// CHECK:STDOUT:   %return.patt.loc7_24.2: @F.%pattern_type.loc7_27 (%pattern_type.51d) = return_slot_pattern %return.param_patt.loc7_27.2, %T.loc7_7.1 [symbolic = %return.patt.loc7_24.2 (constants.%return.patt.b39)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16: <witness> = require_complete_type %C.loc7_21.1 [symbolic = %require_complete.loc7_16 (constants.%require_complete.095)]
// CHECK:STDOUT:   %require_complete.loc7_24: <witness> = require_complete_type %T.loc7_7.1 [symbolic = %require_complete.loc7_24 (constants.%require_complete.944)]
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%C.loc7_21.1 (%C.04a)) -> out %return.param: @F.%T.loc7_7.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_21.1 (%C.04a) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_38.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:     %.loc7_27.1: ref @F.%T.loc7_7.1 (%T) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_7.1 (%T) to %.loc7_27.1 = call %F.specific_fn.loc7_38.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %C.56f) -> out %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.56f = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn.8e2]
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
// CHECK:STDOUT:   %C.loc7_21.1 => constants.%C.04a
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.563
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.6eb
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.41e
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.184
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.51d
// CHECK:STDOUT:   %return.param_patt.loc7_27.2 => constants.%return.param_patt.41d
// CHECK:STDOUT:   %return.patt.loc7_24.2 => constants.%return.patt.b39
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16 => constants.%require_complete.095
// CHECK:STDOUT:   %require_complete.loc7_24 => constants.%require_complete.944
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.c05
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
// CHECK:STDOUT:   %C.loc7_21.1 => constants.%C.56f
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.e3f
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.0b5
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.8c3
// CHECK:STDOUT:   %.loc7_27.2 => constants.%.592
// CHECK:STDOUT:   %pattern_type.loc7_27 => constants.%pattern_type.d8d
// CHECK:STDOUT:   %return.param_patt.loc7_27.2 => constants.%return.param_patt.eae
// CHECK:STDOUT:   %return.patt.loc7_24.2 => constants.%return.patt.e38
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_16 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc7_24 => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.8e2
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
// CHECK:STDOUT:   %I.dcd: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.120: type = pattern_type %I.dcd [symbolic]
// CHECK:STDOUT:   %p.param_patt.38e: %pattern_type.120 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.e65: %pattern_type.120 = wrapper_binding_pattern p, %p.param_patt.38e [symbolic]
// CHECK:STDOUT:   %.a44: Core.Form = init_form %C [concrete]
// CHECK:STDOUT:   %pattern_type.98b: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %return.param_patt: %pattern_type.98b = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.dcd [symbolic]
// CHECK:STDOUT:   %F.specific_fn.c05: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %I.384: type = class_type @I, @I(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.91a: type = pattern_type %I.384 [concrete]
// CHECK:STDOUT:   %p.param_patt.795: %pattern_type.91a = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.4ee: %pattern_type.91a = wrapper_binding_pattern p, %p.param_patt.795 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.c73: <specific function> = specific_function %F, @F(%C) [concrete]
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
// CHECK:STDOUT:     %p.param_patt.loc7_16.1: @F.%pattern_type (%pattern_type.120) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.38e)]
// CHECK:STDOUT:     %p.patt.loc7_16.1: @F.%pattern_type (%pattern_type.120) = wrapper_binding_pattern p, %p.param_patt.loc7_16.1 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.e65)]
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
// CHECK:STDOUT:     %p.param: @F.%I.loc7_21.1 (%I.dcd) = value_param call_param0
// CHECK:STDOUT:     %.loc7_21: type = splice_block %I.loc7_21.2 [symbolic = %I.loc7_21.1 (constants.%I.dcd)] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_7.2 [symbolic = %T.loc7_7.1 (constants.%T)]
// CHECK:STDOUT:       %I.loc7_21.2: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_21.1 (constants.%I.dcd)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%I.loc7_21.1 (%I.dcd) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.91a = value_param_pattern [concrete = constants.%p.param_patt.795]
// CHECK:STDOUT:     %p.patt: %pattern_type.91a = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.4ee]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.98b = out_param_pattern [concrete = constants.%return.param_patt]
// CHECK:STDOUT:     %return.patt: %pattern_type.98b = return_slot_pattern %return.param_patt, %C.ref.loc9_18 [concrete = constants.%return.patt]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc9_18.2: Core.Form = init_form %C.ref.loc9_18 [concrete = constants.%.a44]
// CHECK:STDOUT:     %p.param: %I.384 = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I [concrete = constants.%I.384] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %C.ref.loc9_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %I: type = class_type @I, @I(constants.%C) [concrete = constants.%I.384]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %I.384 = wrapper_binding p, %p.param
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
// CHECK:STDOUT:     .Self = constants.%I.dcd
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
// CHECK:STDOUT:   %I.loc7_21.1: type = class_type @I, @I(%T.loc7_7.1) [symbolic = %I.loc7_21.1 (constants.%I.dcd)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.loc7_21.1 [symbolic = %pattern_type (constants.%pattern_type.120)]
// CHECK:STDOUT:   %p.param_patt.loc7_16.2: @F.%pattern_type (%pattern_type.120) = value_param_pattern [symbolic = %p.param_patt.loc7_16.2 (constants.%p.param_patt.38e)]
// CHECK:STDOUT:   %p.patt.loc7_16.2: @F.%pattern_type (%pattern_type.120) = wrapper_binding_pattern p, %p.param_patt.loc7_16.2 [symbolic = %p.patt.loc7_16.2 (constants.%p.patt.e65)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.loc7_21.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2: <specific function> = specific_function constants.%F, @F(%T.loc7_7.1) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%I.loc7_21.1 (%I.dcd)) -> out %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_21.1 (%I.dcd) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_38.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_38.2 (constants.%F.specific_fn.c05)]
// CHECK:STDOUT:     %.loc7_27.1: ref %C = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init %C to %.loc7_27.1 = call %F.specific_fn.loc7_38.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %I.384) -> out %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.384 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.c73]
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
// CHECK:STDOUT:   %I.loc7_21.1 => constants.%I.dcd
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.120
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.38e
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.e65
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.c05
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
// CHECK:STDOUT:   %I.loc7_21.1 => constants.%I.384
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.91a
// CHECK:STDOUT:   %p.param_patt.loc7_16.2 => constants.%p.param_patt.795
// CHECK:STDOUT:   %p.patt.loc7_16.2 => constants.%p.patt.4ee
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_38.2 => constants.%F.specific_fn.c73
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
// CHECK:STDOUT:   %Outer.651: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U.patt: %pattern_type.9a5 = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.19b: type = parameterized_class_name_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.282: %Inner.type.19b = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.b8a: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %require_complete.868: <witness> = require_complete_type %Outer.651 [symbolic]
// CHECK:STDOUT:   %pattern_type.b11: type = pattern_type %Inner.b8a [symbolic]
// CHECK:STDOUT:   %p.param_patt.a07: %pattern_type.b11 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %p.patt.2ad: %pattern_type.b11 = wrapper_binding_pattern p, %p.param_patt.a07 [symbolic]
// CHECK:STDOUT:   %tuple.type.693: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.fa2: %tuple.type.693 = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.a5e: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %.f18: Core.Form = init_form %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %pattern_type.eee: type = pattern_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %return.param_patt.a41: %pattern_type.eee = out_param_pattern [symbolic]
// CHECK:STDOUT:   %return.patt.453: %pattern_type.eee = return_slot_pattern %return.param_patt.a41, %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.cf8: <witness> = require_complete_type %Inner.b8a [symbolic]
// CHECK:STDOUT:   %require_complete.220: <witness> = require_complete_type %tuple.type.a5e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.57c: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %Outer.1b5: type = class_type @Outer, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.type.178: type = parameterized_class_name_type @Inner, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.generic.787: %Inner.type.178 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.601: type = class_type @Inner, @Inner(%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.49f: type = pattern_type %Inner.601 [concrete]
// CHECK:STDOUT:   %p.param_patt.dcd: %pattern_type.49f = value_param_pattern [concrete]
// CHECK:STDOUT:   %p.patt.558: %pattern_type.49f = wrapper_binding_pattern p, %p.param_patt.dcd [concrete]
// CHECK:STDOUT:   %tuple.919: %tuple.type.693 = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.af1: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %.71f: Core.Form = init_form %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %pattern_type.c66: type = pattern_type %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %return.param_patt.fc8: %pattern_type.c66 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.964: %pattern_type.c66 = return_slot_pattern %return.param_patt.fc8, %tuple.type.af1 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.ed6: <specific function> = specific_function %F, @F(%C, %D) [concrete]
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
// CHECK:STDOUT:     %p.param_patt.loc13_25.1: @F.%pattern_type.loc13_25 (%pattern_type.b11) = value_param_pattern [symbolic = %p.param_patt.loc13_25.2 (constants.%p.param_patt.a07)]
// CHECK:STDOUT:     %p.patt.loc13_25.1: @F.%pattern_type.loc13_25 (%pattern_type.b11) = wrapper_binding_pattern p, %p.param_patt.loc13_25.1 [symbolic = %p.patt.loc13_25.2 (constants.%p.patt.2ad)]
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
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_43.1 (%Inner.b8a) = value_param call_param0
// CHECK:STDOUT:     %.loc13_43: type = splice_block %Inner.loc13_43.2 [symbolic = %Inner.loc13_43.1 (constants.%Inner.b8a)] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref.loc13_33: type = name_ref T, %T.loc13_7.2 [symbolic = %T.loc13_7.1 (constants.%T)]
// CHECK:STDOUT:       %Outer.loc13_34.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_34.1 (constants.%Outer.651)]
// CHECK:STDOUT:       %.loc13_35: @F.%Inner.type (%Inner.type.19b) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %Inner.ref: @F.%Inner.type (%Inner.type.19b) = name_ref Inner, %.loc13_35 [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:       %U.ref.loc13_42: type = name_ref U, %U.loc13_16.2 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_43.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_43.1 (constants.%Inner.b8a)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%Inner.loc13_43.1 (%Inner.b8a) = wrapper_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.a5e) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.a5e) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.param_patt: %pattern_type.49f = value_param_pattern [concrete = constants.%p.param_patt.dcd]
// CHECK:STDOUT:     %p.patt: %pattern_type.49f = wrapper_binding_pattern p, %p.param_patt [concrete = constants.%p.patt.558]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c66 = out_param_pattern [concrete = constants.%return.param_patt.fc8]
// CHECK:STDOUT:     %return.patt: %pattern_type.c66 = return_slot_pattern %return.param_patt, %.loc15_36.3 [concrete = constants.%return.patt.964]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_36.2: %tuple.type.693 = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35) [concrete = constants.%tuple.919]
// CHECK:STDOUT:     %.loc15_36.3: type = converted %.loc15_36.2, constants.%tuple.type.af1 [concrete = constants.%tuple.type.af1]
// CHECK:STDOUT:     %.loc15_36.4: Core.Form = init_form %.loc15_36.3 [concrete = constants.%.71f]
// CHECK:STDOUT:     %p.param: %Inner.601 = value_param call_param0
// CHECK:STDOUT:     %.loc15_25: type = splice_block %Inner [concrete = constants.%Inner.601] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %C.ref.loc15_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Outer: type = class_type @Outer, @Outer(constants.%C) [concrete = constants.%Outer.1b5]
// CHECK:STDOUT:       %.loc15_17: %Inner.type.178 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [concrete = constants.%Inner.generic.787]
// CHECK:STDOUT:       %Inner.ref: %Inner.type.178 = name_ref Inner, %.loc15_17 [concrete = constants.%Inner.generic.787]
// CHECK:STDOUT:       %D.ref.loc15_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [concrete = constants.%Inner.601]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %Inner.601 = wrapper_binding p, %p.param
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
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T.loc4_14.1) [symbolic = %Inner.type (constants.%Inner.type.19b)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.19b) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.19b) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.282)] {
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
// CHECK:STDOUT:     .Self = constants.%Outer.651
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
// CHECK:STDOUT:     .Self = constants.%Inner.b8a
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
// CHECK:STDOUT:   %Outer.loc13_34.1: type = class_type @Outer, @Outer(%T.loc13_7.1) [symbolic = %Outer.loc13_34.1 (constants.%Outer.651)]
// CHECK:STDOUT:   %require_complete.loc13_35: <witness> = require_complete_type %Outer.loc13_34.1 [symbolic = %require_complete.loc13_35 (constants.%require_complete.868)]
// CHECK:STDOUT:   %Inner.type: type = parameterized_class_name_type @Inner, @Outer(%T.loc13_7.1) [symbolic = %Inner.type (constants.%Inner.type.19b)]
// CHECK:STDOUT:   %Inner.generic: @F.%Inner.type (%Inner.type.19b) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.282)]
// CHECK:STDOUT:   %Inner.loc13_43.1: type = class_type @Inner, @Inner(%T.loc13_7.1, %U.loc13_16.1) [symbolic = %Inner.loc13_43.1 (constants.%Inner.b8a)]
// CHECK:STDOUT:   %pattern_type.loc13_25: type = pattern_type %Inner.loc13_43.1 [symbolic = %pattern_type.loc13_25 (constants.%pattern_type.b11)]
// CHECK:STDOUT:   %p.param_patt.loc13_25.2: @F.%pattern_type.loc13_25 (%pattern_type.b11) = value_param_pattern [symbolic = %p.param_patt.loc13_25.2 (constants.%p.param_patt.a07)]
// CHECK:STDOUT:   %p.patt.loc13_25.2: @F.%pattern_type.loc13_25 (%pattern_type.b11) = wrapper_binding_pattern p, %p.param_patt.loc13_25.2 [symbolic = %p.patt.loc13_25.2 (constants.%p.patt.2ad)]
// CHECK:STDOUT:   %tuple: %tuple.type.693 = tuple_value (%T.loc13_7.1, %U.loc13_16.1) [symbolic = %tuple (constants.%tuple.fa2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc13_7.1, %U.loc13_16.1) [symbolic = %tuple.type (constants.%tuple.type.a5e)]
// CHECK:STDOUT:   %.loc13_54.2: Core.Form = init_form %tuple.type [symbolic = %.loc13_54.2 (constants.%.f18)]
// CHECK:STDOUT:   %pattern_type.loc13_54: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13_54 (constants.%pattern_type.eee)]
// CHECK:STDOUT:   %return.param_patt.loc13_54.2: @F.%pattern_type.loc13_54 (%pattern_type.eee) = out_param_pattern [symbolic = %return.param_patt.loc13_54.2 (constants.%return.param_patt.a41)]
// CHECK:STDOUT:   %return.patt.loc13_46.2: @F.%pattern_type.loc13_54 (%pattern_type.eee) = return_slot_pattern %return.param_patt.loc13_54.2, %tuple.type [symbolic = %return.patt.loc13_46.2 (constants.%return.patt.453)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_25: <witness> = require_complete_type %Inner.loc13_43.1 [symbolic = %require_complete.loc13_25 (constants.%require_complete.cf8)]
// CHECK:STDOUT:   %require_complete.loc13_46: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_46 (constants.%require_complete.220)]
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2: <specific function> = specific_function constants.%F, @F(%T.loc13_7.1, %U.loc13_16.1) [symbolic = %F.specific_fn.loc13_65.2 (constants.%F.specific_fn.57c)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%Inner.loc13_43.1 (%Inner.b8a)) -> out %return.param: @F.%tuple.type (%tuple.type.a5e) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_43.1 (%Inner.b8a) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc13_65.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc13_65.2 (constants.%F.specific_fn.57c)]
// CHECK:STDOUT:     %.loc13_54.1: ref @F.%tuple.type (%tuple.type.a5e) = splice_block %return.param {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.a5e) to %.loc13_54.1 = call %F.specific_fn.loc13_65.1(%p.ref)
// CHECK:STDOUT:     return %F.call to %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %Inner.601) -> out %return.param: %tuple.type.af1 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.601 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.ed6]
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
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.19b
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.282
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
// CHECK:STDOUT:   %Outer.loc13_34.1 => constants.%Outer.651
// CHECK:STDOUT:   %require_complete.loc13_35 => constants.%require_complete.868
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.19b
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.282
// CHECK:STDOUT:   %Inner.loc13_43.1 => constants.%Inner.b8a
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.b11
// CHECK:STDOUT:   %p.param_patt.loc13_25.2 => constants.%p.param_patt.a07
// CHECK:STDOUT:   %p.patt.loc13_25.2 => constants.%p.patt.2ad
// CHECK:STDOUT:   %tuple => constants.%tuple.fa2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.a5e
// CHECK:STDOUT:   %.loc13_54.2 => constants.%.f18
// CHECK:STDOUT:   %pattern_type.loc13_54 => constants.%pattern_type.eee
// CHECK:STDOUT:   %return.param_patt.loc13_54.2 => constants.%return.param_patt.a41
// CHECK:STDOUT:   %return.patt.loc13_46.2 => constants.%return.patt.453
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_25 => constants.%require_complete.cf8
// CHECK:STDOUT:   %require_complete.loc13_46 => constants.%require_complete.220
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2 => constants.%F.specific_fn.57c
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.patt.loc4_14.2 => constants.%T.patt
// CHECK:STDOUT:   %T.loc4_14.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.178
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.787
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
// CHECK:STDOUT:   %Outer.loc13_34.1 => constants.%Outer.1b5
// CHECK:STDOUT:   %require_complete.loc13_35 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.178
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.787
// CHECK:STDOUT:   %Inner.loc13_43.1 => constants.%Inner.601
// CHECK:STDOUT:   %pattern_type.loc13_25 => constants.%pattern_type.49f
// CHECK:STDOUT:   %p.param_patt.loc13_25.2 => constants.%p.param_patt.dcd
// CHECK:STDOUT:   %p.patt.loc13_25.2 => constants.%p.patt.558
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
// CHECK:STDOUT:   %F.specific_fn.loc13_65.2 => constants.%F.specific_fn.ed6
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
// CHECK:STDOUT:   %pattern_type.d94: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %N.patt.d64: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %N.c34: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %WithNontype.type: type = parameterized_class_name_type @WithNontype [concrete]
// CHECK:STDOUT:   %WithNontype.generic: %WithNontype.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithNontype.5ab: type = class_type @WithNontype, @WithNontype(%N.c34) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.6c9: type = pattern_type %WithNontype.5ab [symbolic]
// CHECK:STDOUT:   %x.param_patt.8a9: %pattern_type.6c9 = value_param_pattern [symbolic]
// CHECK:STDOUT:   %x.patt.a70: %pattern_type.6c9 = wrapper_binding_pattern x, %x.param_patt.8a9 [symbolic]
// CHECK:STDOUT:   %.0b3: Core.Form = init_form %i32 [concrete]
// CHECK:STDOUT:   %return.param_patt.299: %pattern_type.d94 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.6fa: %pattern_type.d94 = return_slot_pattern %return.param_patt.299, %i32 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4fe: <witness> = require_complete_type %WithNontype.5ab [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.a1d: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.fe9) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.3c1: %Int.as.Copy.impl.Op.type.a1d = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.8af: <witness> = impl_witness imports.%Copy.impl_witness_table.f44, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.8af) [concrete]
// CHECK:STDOUT:   %Copy.WithSelf.Op.type.f76: type = fn_type @Copy.WithSelf.Op, @Copy.WithSelf(%Copy.facet) [concrete]
// CHECK:STDOUT:   %.2b9: type = fn_type_with_self_type %Copy.WithSelf.Op.type.f76, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.648: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.cae: %Int.as.Copy.impl.Op.type.648 = struct_value () [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.ed4: <bound method> = bound_method %N.c34, %Int.as.Copy.impl.Op.cae [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.cae, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.a53: <bound method> = bound_method %N.c34, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.0ff: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.0ff = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.6b0: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.d73: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.f53, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.6b0 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.d73) [concrete]
// CHECK:STDOUT:   %ImplicitAs.WithSelf.Convert.type.58f: type = fn_type @ImplicitAs.WithSelf.Convert, @ImplicitAs.WithSelf(%i32, %ImplicitAs.facet) [concrete]
// CHECK:STDOUT:   %.33a: type = fn_type_with_self_type %ImplicitAs.WithSelf.Convert.type.58f, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.766 = struct_value () [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60 [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.f60, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.036: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.753: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %WithNontype.e5d: type = class_type @WithNontype, @WithNontype(%int_0.753) [concrete]
// CHECK:STDOUT:   %WithNontype.val: %WithNontype.e5d = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.832: type = pattern_type %WithNontype.e5d [concrete]
// CHECK:STDOUT:   %x.param_patt.b39: %pattern_type.832 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.c6a: %pattern_type.832 = wrapper_binding_pattern x, %x.param_patt.b39 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_0.753) [concrete]
// CHECK:STDOUT:   %.f5e: ref %WithNontype.e5d = temporary invalid, %WithNontype.val [concrete]
// CHECK:STDOUT:   %SelfDestruct.type: type = fn_type @SelfDestruct [concrete]
// CHECK:STDOUT:   %SelfDestruct: %SelfDestruct.type = struct_value () [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.type: type = facet_type <@SubobjectDestroy> [concrete]
// CHECK:STDOUT:   %T.09f: %SubobjectDestroy.type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.type.dea: type = fn_type @T.as_type.as.DestroyImpl.impl.Op, @T.as_type.as.DestroyImpl.impl(%T.09f) [symbolic]
// CHECK:STDOUT:   %T.as_type.as.DestroyImpl.impl.Op.6e5: %T.as_type.as.DestroyImpl.impl.Op.type.dea = struct_value () [symbolic]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.type.6870fd.2: type = fn_type @SubobjectDestroy.WithSelf.Op.loc9_15.2 [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.WithSelf.Op.30f1ee.2: %SubobjectDestroy.WithSelf.Op.type.6870fd.2 = struct_value () [concrete]
// CHECK:STDOUT:   %custom_witness.b80ef6.2: <witness> = custom_witness (%SubobjectDestroy.WithSelf.Op.30f1ee.2), @SubobjectDestroy [concrete]
// CHECK:STDOUT:   %SubobjectDestroy.facet.7f6: %SubobjectDestroy.type = facet_value %WithNontype.e5d, (%custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %DestroyImpl.impl_witness.405: <witness> = impl_witness imports.%DestroyImpl.impl_witness_table, @T.as_type.as.DestroyImpl.impl(%SubobjectDestroy.facet.7f6) [concrete]
// CHECK:STDOUT:   %Destroy.facet: %Destroy.type = facet_value %WithNontype.e5d, (%DestroyImpl.impl_witness.405, %custom_witness.b80ef6.2) [concrete]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function %SelfDestruct, @SelfDestruct(%Destroy.facet) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.4fb: <bound method> = bound_method %int_0.753, %Int.as.Copy.impl.Op.cae [concrete]
// CHECK:STDOUT:   %bound_method.149: <bound method> = bound_method %int_0.753, %Int.as.Copy.impl.Op.specific_fn [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .Copy = %Core.Copy
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     .SelfDestruct = %Core.SelfDestruct
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/parts/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.Copy: type = import_ref Core//prelude/parts/copy, Copy, loaded [concrete = constants.%Copy.type]
// CHECK:STDOUT:   %Core.import_ref.72f: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.a1d) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.3c1)]
// CHECK:STDOUT:   %Copy.impl_witness_table.f44 = impl_witness_table (%Core.import_ref.72f), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.0ff = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e47: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.3c6) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.6f2)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.f53 = impl_witness_table (%Core.import_ref.e47), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
// CHECK:STDOUT:   %Core.SelfDestruct: %SelfDestruct.type = import_ref Core//prelude/parts/destroy, SelfDestruct, loaded [concrete = constants.%SelfDestruct]
// CHECK:STDOUT:   %Core.import_ref.74d: @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op.type (%T.as_type.as.DestroyImpl.impl.Op.type.dea) = import_ref Core//prelude/parts/destroy, loc{{\d+_\d+}}, loaded [symbolic = @T.as_type.as.DestroyImpl.impl.%T.as_type.as.DestroyImpl.impl.Op (constants.%T.as_type.as.DestroyImpl.impl.Op.6e5)]
// CHECK:STDOUT:   %DestroyImpl.impl_witness_table = impl_witness_table (%Core.import_ref.74d), @T.as_type.as.DestroyImpl.impl [concrete]
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
// CHECK:STDOUT:     %N.patt.loc4_20.1: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_20.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.c34)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc6_7.1: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:     %x.param_patt.loc6_22.1: @F.%pattern_type (%pattern_type.6c9) = value_param_pattern [symbolic = %x.param_patt.loc6_22.2 (constants.%x.param_patt.8a9)]
// CHECK:STDOUT:     %x.patt.loc6_22.1: @F.%pattern_type (%pattern_type.6c9) = wrapper_binding_pattern x, %x.param_patt.loc6_22.1 [symbolic = %x.patt.loc6_22.2 (constants.%x.patt.a70)]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32.loc6_43 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32.loc6_43: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_43: Core.Form = init_form %i32.loc6_43 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %.loc6_9: type = splice_block %i32.loc6_9 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self.frozen: type = symbolic_binding .Self [symbolic_self = constants.%.Self.frozen]
// CHECK:STDOUT:       %i32.loc6_9: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_7.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.c34)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_37.1 (%WithNontype.5ab) = value_param call_param0
// CHECK:STDOUT:     %.loc6_37: type = splice_block %WithNontype.loc6_37.2 [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.5ab)] {
// CHECK:STDOUT:       %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:       %N.ref.loc6_36: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.c34)]
// CHECK:STDOUT:       %WithNontype.loc6_37.2: type = class_type @WithNontype, @WithNontype(constants.%N.c34) [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.5ab)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_37.1 (%WithNontype.5ab) = wrapper_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.d94 = out_param_pattern [concrete = constants.%return.param_patt.299]
// CHECK:STDOUT:     %return.patt: %pattern_type.d94 = return_slot_pattern %return.param_patt, %i32 [concrete = constants.%return.patt.6fa]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc8: Core.Form = init_form %i32 [concrete = constants.%.0b3]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_20.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc4_20.2: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_20.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   %N.loc4_20.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_20.1 (constants.%N.c34)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.5ab
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_7.2: %i32) {
// CHECK:STDOUT:   %N.patt.loc6_7.2: %pattern_type.d94 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_7.2 (constants.%N.patt.d64)]
// CHECK:STDOUT:   %N.loc6_7.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_7.1 (constants.%N.c34)]
// CHECK:STDOUT:   %WithNontype.loc6_37.1: type = class_type @WithNontype, @WithNontype(%N.loc6_7.1) [symbolic = %WithNontype.loc6_37.1 (constants.%WithNontype.5ab)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %WithNontype.loc6_37.1 [symbolic = %pattern_type (constants.%pattern_type.6c9)]
// CHECK:STDOUT:   %x.param_patt.loc6_22.2: @F.%pattern_type (%pattern_type.6c9) = value_param_pattern [symbolic = %x.param_patt.loc6_22.2 (constants.%x.param_patt.8a9)]
// CHECK:STDOUT:   %x.patt.loc6_22.2: @F.%pattern_type (%pattern_type.6c9) = wrapper_binding_pattern x, %x.param_patt.loc6_22.2 [symbolic = %x.patt.loc6_22.2 (constants.%x.patt.a70)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WithNontype.loc6_37.1 [symbolic = %require_complete (constants.%require_complete.4fe)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.cae [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.ed4)]
// CHECK:STDOUT:   %bound_method.loc6_56.3: <bound method> = bound_method %N.loc6_7.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_56.3 (constants.%bound_method.a53)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%WithNontype.loc6_37.1 (%WithNontype.5ab)) -> out %return.param: %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_56: %i32 = name_ref N, %N.loc6_7.2 [symbolic = %N.loc6_7.1 (constants.%N.c34)]
// CHECK:STDOUT:     %impl.elem0: %.2b9 = impl_witness_access constants.%Copy.impl_witness.8af, element0 [concrete = constants.%Int.as.Copy.impl.Op.cae]
// CHECK:STDOUT:     %bound_method.loc6_56.1: <bound method> = bound_method %N.ref.loc6_56, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.ed4)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_56.2: <bound method> = bound_method %N.ref.loc6_56, %specific_fn [symbolic = %bound_method.loc6_56.3 (constants.%bound_method.a53)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_56.2(%N.ref.loc6_56) [symbolic = %N.loc6_7.1 (constants.%N.c34)]
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
// CHECK:STDOUT:   %impl.elem0: %.33a = impl_witness_access constants.%ImplicitAs.impl_witness.d73, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.f60]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.036]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_31.2(%int_0) [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc9_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %.loc9_31.2: %i32 = converted %int_0, %.loc9_31.1 [concrete = constants.%int_0.753]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%int_0.753) [concrete = constants.%WithNontype.e5d]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.e5d = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.e5d to %.loc9_13.2 = class_init () [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %.loc9_15.1: init %WithNontype.e5d = converted %.loc9_13.1, %.loc9_13.3 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_0.753) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc9_15.2: ref %WithNontype.e5d = temporary %.loc9_13.2, %.loc9_15.1 [concrete = constants.%.f5e]
// CHECK:STDOUT:   %.loc9_15.3: %WithNontype.e5d = acquire_value %.loc9_15.2 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc9_15.3)
// CHECK:STDOUT:   %Destroy.facet.loc9_15.1: %Destroy.type = facet_value constants.%WithNontype.e5d, (constants.%DestroyImpl.impl_witness.405, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_15.4: %Destroy.type = converted constants.%WithNontype.e5d, %Destroy.facet.loc9_15.1 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %Destroy.facet.loc9_15.2: %Destroy.type = facet_value constants.%WithNontype.e5d, (constants.%DestroyImpl.impl_witness.405, constants.%custom_witness.b80ef6.2) [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %.loc9_15.5: %Destroy.type = converted constants.%WithNontype.e5d, %Destroy.facet.loc9_15.2 [concrete = constants.%Destroy.facet]
// CHECK:STDOUT:   %SelfDestruct.specific_fn: <specific function> = specific_function constants.%SelfDestruct, @SelfDestruct(constants.%Destroy.facet) [concrete = constants.%SelfDestruct.specific_fn]
// CHECK:STDOUT:   %SelfDestruct.call: init %empty_tuple.type = call %SelfDestruct.specific_fn(%.loc9_15.2)
// CHECK:STDOUT:   return %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_15.1(%self.param: ref %empty_struct_type) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: fn @SubobjectDestroy.WithSelf.Op.loc9_15.2(%self.param: ref %WithNontype.e5d) = "no_op";
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%N.c34
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.c34) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%N.c34
// CHECK:STDOUT:   %WithNontype.loc6_37.1 => constants.%WithNontype.5ab
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.6c9
// CHECK:STDOUT:   %x.param_patt.loc6_22.2 => constants.%x.param_patt.8a9
// CHECK:STDOUT:   %x.patt.loc6_22.2 => constants.%x.patt.a70
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%int_0.753) {
// CHECK:STDOUT:   %N.patt.loc4_20.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc4_20.1 => constants.%int_0.753
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_0.753) {
// CHECK:STDOUT:   %N.patt.loc6_7.2 => constants.%N.patt.d64
// CHECK:STDOUT:   %N.loc6_7.1 => constants.%int_0.753
// CHECK:STDOUT:   %WithNontype.loc6_37.1 => constants.%WithNontype.e5d
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.832
// CHECK:STDOUT:   %x.param_patt.loc6_22.2 => constants.%x.param_patt.b39
// CHECK:STDOUT:   %x.patt.loc6_22.2 => constants.%x.patt.c6a
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.4fb
// CHECK:STDOUT:   %bound_method.loc6_56.3 => constants.%bound_method.149
// CHECK:STDOUT: }
// CHECK:STDOUT:
