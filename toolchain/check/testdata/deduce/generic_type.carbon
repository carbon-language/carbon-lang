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

fn F[N:! i32](x: WithNontype(N)) -> i32 { return N; }

fn G() -> i32 {
  return F({} as WithNontype(0));
}

// CHECK:STDOUT: --- class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.3f0: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.06f: type = pattern_type %C.3f0 [symbolic]
// CHECK:STDOUT:   %pattern_type.e68: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4b7: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.c98: <witness> = require_complete_type %C.3f0 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.96e: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %C.131: type = class_type @C, @C(%D) [concrete]
// CHECK:STDOUT:   %pattern_type.7f8: type = pattern_type %C.131 [concrete]
// CHECK:STDOUT:   %pattern_type.510: type = pattern_type %D [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.c4a: <specific function> = specific_function %F, @F(%D) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc7_16 (%pattern_type.06f) = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc7_16 (%pattern_type.06f) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_25 (%pattern_type.e68) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_25 (%pattern_type.e68) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_28: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_22.1 (%C.3f0) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %C.loc7_22.2 [symbolic = %C.loc7_22.1 (constants.%C.3f0)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc7_21: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_22.2: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_22.1 (constants.%C.3f0)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%C.loc7_22.1 (%C.3f0) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.1 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.1 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.7f8 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.7f8 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.510 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.510 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %p.param: %C.131 = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %C [concrete = constants.%C.131] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %D.ref.loc9_11: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %C: type = class_type @C, @C(constants.%D) [concrete = constants.%C.131]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %C.131 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.3f0
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
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: type) {
// CHECK:STDOUT:   %T.loc7_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %C.loc7_22.1: type = class_type @C, @C(%T.loc7_6.1) [symbolic = %C.loc7_22.1 (constants.%C.3f0)]
// CHECK:STDOUT:   %pattern_type.loc7_16: type = pattern_type %C.loc7_22.1 [symbolic = %pattern_type.loc7_16 (constants.%pattern_type.06f)]
// CHECK:STDOUT:   %pattern_type.loc7_25: type = pattern_type %T.loc7_6.1 [symbolic = %pattern_type.loc7_25 (constants.%pattern_type.e68)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_25: <witness> = require_complete_type %T.loc7_6.1 [symbolic = %require_complete.loc7_25 (constants.%require_complete.4b7)]
// CHECK:STDOUT:   %require_complete.loc7_17: <witness> = require_complete_type %C.loc7_22.1 [symbolic = %require_complete.loc7_17 (constants.%require_complete.c98)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.1) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.96e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%C.loc7_22.1 (%C.3f0)) -> %return.param: @F.%T.loc7_6.1 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_22.1 (%C.3f0) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.96e)]
// CHECK:STDOUT:     %.loc7_25: ref @F.%T.loc7_6.1 (%T) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_6.1 (%T) = call %F.specific_fn.loc7_39.1(%p.ref) to %.loc7_25
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %C.131) -> %return.param: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.131 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%D) [concrete = constants.%F.specific_fn.c4a]
// CHECK:STDOUT:   %.loc9_15: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %F.specific_fn(%p.ref) to %.loc9_15
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %C.loc7_22.1 => constants.%C.3f0
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.06f
// CHECK:STDOUT:   %pattern_type.loc7_25 => constants.%pattern_type.e68
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_25 => constants.%require_complete.4b7
// CHECK:STDOUT:   %require_complete.loc7_17 => constants.%require_complete.c98
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.96e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%D
// CHECK:STDOUT:   %C.loc7_22.1 => constants.%C.131
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.7f8
// CHECK:STDOUT:   %pattern_type.loc7_25 => constants.%pattern_type.510
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_25 => constants.%complete_type
// CHECK:STDOUT:   %require_complete.loc7_17 => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.c4a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type: type = generic_class_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.b0e: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.0b8: type = pattern_type %I.b0e [symbolic]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.b0e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.96e: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %I.ed8: type = class_type @I, @I(%C) [concrete]
// CHECK:STDOUT:   %pattern_type.917: type = pattern_type %I.ed8 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.04a: <specific function> = specific_function %F, @F(%C) [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_9.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type (%pattern_type.0b8) = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type (%pattern_type.0b8) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc7_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_22.1 (%I.b0e) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %I.loc7_22.2 [symbolic = %I.loc7_22.1 (constants.%I.b0e)] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_6.2 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:       %I.loc7_22.2: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_22.1 (constants.%I.b0e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%I.loc7_22.1 (%I.b0e) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.917 = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.917 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %p.param: %I.ed8 = value_param call_param0
// CHECK:STDOUT:     %.loc9_12: type = splice_block %I [concrete = constants.%I.ed8] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %C.ref.loc9_11: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %I: type = class_type @I, @I(constants.%C) [concrete = constants.%I.ed8]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %I.ed8 = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.b0e
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
// CHECK:STDOUT: generic fn @F(%T.loc7_6.2: type) {
// CHECK:STDOUT:   %T.loc7_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc7_6.1 (constants.%T)]
// CHECK:STDOUT:   %I.loc7_22.1: type = class_type @I, @I(%T.loc7_6.1) [symbolic = %I.loc7_22.1 (constants.%I.b0e)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.loc7_22.1 [symbolic = %pattern_type (constants.%pattern_type.0b8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.loc7_22.1 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.1) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.96e)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%I.loc7_22.1 (%I.b0e)) -> %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_22.1 (%I.b0e) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.96e)]
// CHECK:STDOUT:     %.loc7_25: ref %C = splice_block %return {}
// CHECK:STDOUT:     %F.call: init %C = call %F.specific_fn.loc7_39.1(%p.ref) to %.loc7_25
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %I.ed8) -> %return.param: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.ed8 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C) [concrete = constants.%F.specific_fn.04a]
// CHECK:STDOUT:   %.loc9_15: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.specific_fn(%p.ref) to %.loc9_15
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%T
// CHECK:STDOUT:   %I.loc7_22.1 => constants.%I.b0e
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.0b8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.96e
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc7_6.1 => constants.%C
// CHECK:STDOUT:   %I.loc7_22.1 => constants.%I.ed8
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.917
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.04a
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %type: type = facet_type <type> [concrete]
// CHECK:STDOUT:   %.Self: %type = symbolic_binding .Self [symbolic_self]
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.d8f: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = symbolic_binding U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.e10: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.bcd: %Inner.type.e10 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.f6c: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %require_complete.799: <witness> = require_complete_type %Outer.d8f [symbolic]
// CHECK:STDOUT:   %pattern_type.a25: type = pattern_type %Inner.f6c [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.225: %tuple.type.24b = tuple_value (%T, %U) [symbolic]
// CHECK:STDOUT:   %tuple.type.d18: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.b8b: type = pattern_type %tuple.type.d18 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fdf: <witness> = require_complete_type %tuple.type.d18 [symbolic]
// CHECK:STDOUT:   %require_complete.319: <witness> = require_complete_type %Inner.f6c [symbolic]
// CHECK:STDOUT:   %F.specific_fn.b44: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %Outer.7c4: type = class_type @Outer, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.type.181: type = generic_class_type @Inner, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.generic.205: %Inner.type.181 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.bbd: type = class_type @Inner, @Inner(%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.83c: type = pattern_type %Inner.bbd [concrete]
// CHECK:STDOUT:   %tuple.f3a: %tuple.type.24b = tuple_value (%C, %D) [concrete]
// CHECK:STDOUT:   %tuple.type.e8a: type = tuple_type (%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.9ec: type = pattern_type %tuple.type.e8a [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %F.specific_fn.4a7: <specific function> = specific_function %F, @F(%C, %D) [concrete]
// CHECK:STDOUT:   %complete_type.53b: <witness> = complete_type_witness %tuple.type.e8a [concrete]
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
// CHECK:STDOUT:     %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc4_13.2: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc13_26 (%pattern_type.a25) = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc13_26 (%pattern_type.a25) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc13_48 (%pattern_type.b8b) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc13_48 (%pattern_type.b8b) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_52: type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_55: type = name_ref U, %U.loc13_16.2 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:     %.loc13_56.1: %tuple.type.24b = tuple_literal (%T.ref.loc13_52, %U.ref.loc13_55) [symbolic = %tuple (constants.%tuple.225)]
// CHECK:STDOUT:     %.loc13_56.2: type = converted %.loc13_56.1, constants.%tuple.type.d18 [symbolic = %tuple.type (constants.%tuple.type.d18)]
// CHECK:STDOUT:     %.Self.1: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %T.loc13_6.2: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:     %.Self.2: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:     %U.loc13_16.2: type = symbolic_binding U, 1 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_45.1 (%Inner.f6c) = value_param call_param0
// CHECK:STDOUT:     %.loc13_45: type = splice_block %Inner.loc13_45.2 [symbolic = %Inner.loc13_45.1 (constants.%Inner.f6c)] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref.loc13_35: type = name_ref T, %T.loc13_6.2 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:       %Outer.loc13_36.2: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_36.1 (constants.%Outer.d8f)]
// CHECK:STDOUT:       %.loc13_37: @F.%Inner.type (%Inner.type.e10) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.bcd)]
// CHECK:STDOUT:       %Inner.ref: @F.%Inner.type (%Inner.type.e10) = name_ref Inner, %.loc13_37 [symbolic = %Inner.generic (constants.%Inner.generic.bcd)]
// CHECK:STDOUT:       %U.ref.loc13_44: type = name_ref U, %U.loc13_16.2 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_45.2: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_45.1 (constants.%Inner.f6c)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%Inner.loc13_45.1 (%Inner.f6c) = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.d18) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.d18) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.83c = value_binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.83c = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ec = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ec = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_36.1: %tuple.type.24b = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35) [concrete = constants.%tuple.f3a]
// CHECK:STDOUT:     %.loc15_36.2: type = converted %.loc15_36.1, constants.%tuple.type.e8a [concrete = constants.%tuple.type.e8a]
// CHECK:STDOUT:     %p.param: %Inner.bbd = value_param call_param0
// CHECK:STDOUT:     %.loc15_25: type = splice_block %Inner [concrete = constants.%Inner.bbd] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %C.ref.loc15_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Outer: type = class_type @Outer, @Outer(constants.%C) [concrete = constants.%Outer.7c4]
// CHECK:STDOUT:       %.loc15_17: %Inner.type.181 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [concrete = constants.%Inner.generic.205]
// CHECK:STDOUT:       %Inner.ref: %Inner.type.181 = name_ref Inner, %.loc15_17 [concrete = constants.%Inner.generic.205]
// CHECK:STDOUT:       %D.ref.loc15_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [concrete = constants.%Inner.bbd]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %Inner.bbd = value_binding p, %p.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.e8a = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.e8a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = symbolic_binding T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.1) [symbolic = %Inner.type (constants.%Inner.type.e10)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.e10) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.bcd)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.e10) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.bcd)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %U.loc5_15.2: type = symbolic_binding U, 1 [symbolic = %U.loc5_15.1 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.d8f
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.2: type, %U.loc5_15.2: type) {
// CHECK:STDOUT:   %U.loc5_15.1: type = symbolic_binding U, 1 [symbolic = %U.loc5_15.1 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.f6c
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
// CHECK:STDOUT: generic fn @F(%T.loc13_6.2: type, %U.loc13_16.2: type) {
// CHECK:STDOUT:   %T.loc13_6.1: type = symbolic_binding T, 0 [symbolic = %T.loc13_6.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc13_16.1: type = symbolic_binding U, 1 [symbolic = %U.loc13_16.1 (constants.%U)]
// CHECK:STDOUT:   %Outer.loc13_36.1: type = class_type @Outer, @Outer(%T.loc13_6.1) [symbolic = %Outer.loc13_36.1 (constants.%Outer.d8f)]
// CHECK:STDOUT:   %require_complete.loc13_37: <witness> = require_complete_type %Outer.loc13_36.1 [symbolic = %require_complete.loc13_37 (constants.%require_complete.799)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc13_6.1) [symbolic = %Inner.type (constants.%Inner.type.e10)]
// CHECK:STDOUT:   %Inner.generic: @F.%Inner.type (%Inner.type.e10) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.bcd)]
// CHECK:STDOUT:   %Inner.loc13_45.1: type = class_type @Inner, @Inner(%T.loc13_6.1, %U.loc13_16.1) [symbolic = %Inner.loc13_45.1 (constants.%Inner.f6c)]
// CHECK:STDOUT:   %pattern_type.loc13_26: type = pattern_type %Inner.loc13_45.1 [symbolic = %pattern_type.loc13_26 (constants.%pattern_type.a25)]
// CHECK:STDOUT:   %tuple: %tuple.type.24b = tuple_value (%T.loc13_6.1, %U.loc13_16.1) [symbolic = %tuple (constants.%tuple.225)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc13_6.1, %U.loc13_16.1) [symbolic = %tuple.type (constants.%tuple.type.d18)]
// CHECK:STDOUT:   %pattern_type.loc13_48: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13_48 (constants.%pattern_type.b8b)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_48 (constants.%require_complete.fdf)]
// CHECK:STDOUT:   %require_complete.loc13_27: <witness> = require_complete_type %Inner.loc13_45.1 [symbolic = %require_complete.loc13_27 (constants.%require_complete.319)]
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2: <specific function> = specific_function constants.%F, @F(%T.loc13_6.1, %U.loc13_16.1) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.b44)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%Inner.loc13_45.1 (%Inner.f6c)) -> %return.param: @F.%tuple.type (%tuple.type.d18) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_45.1 (%Inner.f6c) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc13_67.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.b44)]
// CHECK:STDOUT:     %.loc13_48: ref @F.%tuple.type (%tuple.type.d18) = splice_block %return {}
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.d18) = call %F.specific_fn.loc13_67.1(%p.ref) to %.loc13_48
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %Inner.bbd) -> %return.param: %tuple.type.e8a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.bbd = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.4a7]
// CHECK:STDOUT:   %.loc15_28: ref %tuple.type.e8a = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %tuple.type.e8a = call %F.specific_fn(%p.ref) to %.loc15_28
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e10
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.bcd
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.1 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_36.1 => constants.%Outer.d8f
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%require_complete.799
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.e10
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.bcd
// CHECK:STDOUT:   %Inner.loc13_45.1 => constants.%Inner.f6c
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.a25
// CHECK:STDOUT:   %tuple => constants.%tuple.225
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.d18
// CHECK:STDOUT:   %pattern_type.loc13_48 => constants.%pattern_type.b8b
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%require_complete.fdf
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%require_complete.319
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.b44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.181
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%C, constants.%D) {
// CHECK:STDOUT:   %U.loc5_15.1 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc13_6.1 => constants.%C
// CHECK:STDOUT:   %U.loc13_16.1 => constants.%D
// CHECK:STDOUT:   %Outer.loc13_36.1 => constants.%Outer.7c4
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.181
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.205
// CHECK:STDOUT:   %Inner.loc13_45.1 => constants.%Inner.bbd
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.83c
// CHECK:STDOUT:   %tuple => constants.%tuple.f3a
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e8a
// CHECK:STDOUT:   %pattern_type.loc13_48 => constants.%pattern_type.9ec
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%complete_type.53b
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%complete_type.357
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.4a7
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
// CHECK:STDOUT:   %N.586: Core.IntLiteral = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.fe8: %i32 = symbolic_binding N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %WithNontype.type: type = generic_class_type @WithNontype [concrete]
// CHECK:STDOUT:   %WithNontype.generic: %WithNontype.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithNontype.e02: type = class_type @WithNontype, @WithNontype(%N.fe8) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.052: type = pattern_type %WithNontype.e02 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9ca: <witness> = require_complete_type %WithNontype.e02 [symbolic]
// CHECK:STDOUT:   %Copy.type: type = facet_type <@Copy> [concrete]
// CHECK:STDOUT:   %Copy.Op.type: type = fn_type @Copy.Op [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.24b: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%N.586) [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.c95: %Int.as.Copy.impl.Op.type.24b = struct_value () [symbolic]
// CHECK:STDOUT:   %Copy.impl_witness.fb7: <witness> = impl_witness imports.%Copy.impl_witness_table.b6a, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.type.469: type = fn_type @Int.as.Copy.impl.Op, @Int.as.Copy.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.dfd: %Int.as.Copy.impl.Op.type.469 = struct_value () [concrete]
// CHECK:STDOUT:   %Copy.facet: %Copy.type = facet_value %i32, (%Copy.impl_witness.fb7) [concrete]
// CHECK:STDOUT:   %.65f: type = fn_type_with_self_type %Copy.Op.type, %Copy.facet [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.691: <bound method> = bound_method %N.fe8, %Int.as.Copy.impl.Op.dfd [symbolic]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.specific_fn: <specific function> = specific_function %Int.as.Copy.impl.Op.dfd, @Int.as.Copy.impl.Op(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.f2b: <bound method> = bound_method %N.fe8, %Int.as.Copy.impl.Op.specific_fn [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.d14: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %ImplicitAs.Convert.type.1b6: type = fn_type @ImplicitAs.Convert, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To: Core.IntLiteral = symbolic_binding To, 0 [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%To) [symbolic]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.bc9: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.132, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e: type = fn_type @Core.IntLiteral.as.ImplicitAs.impl.Convert, @Core.IntLiteral.as.ImplicitAs.impl(%int_32) [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b: %Core.IntLiteral.as.ImplicitAs.impl.Convert.type.51e = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.d14 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.bc9) [concrete]
// CHECK:STDOUT:   %.322: type = fn_type_with_self_type %ImplicitAs.Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.bound: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b [concrete]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn: <specific function> = specific_function %Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b, @Core.IntLiteral.as.ImplicitAs.impl.Convert(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method.9be: <bound method> = bound_method %int_0.5c6, %Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %WithNontype.b82: type = class_type @WithNontype, @WithNontype(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %WithNontype.val: %WithNontype.b82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b66: type = pattern_type %WithNontype.b82 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %Destroy.type: type = facet_type <@Destroy> [concrete]
// CHECK:STDOUT:   %type_where: type = facet_type <type where .Self impls <CanDestroy>> [concrete]
// CHECK:STDOUT:   %facet_value: %type_where = facet_value %WithNontype.b82, () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.type.819: type = fn_type @DestroyT.binding.as_type.as.Destroy.impl.Op, @DestroyT.binding.as_type.as.Destroy.impl(%facet_value) [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.fc3: %DestroyT.binding.as_type.as.Destroy.impl.Op.type.819 = struct_value () [concrete]
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function %DestroyT.binding.as_type.as.Destroy.impl.Op.fc3, @DestroyT.binding.as_type.as.Destroy.impl.Op(%facet_value) [concrete]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound.986: <bound method> = bound_method %int_0.6a9, %Int.as.Copy.impl.Op.dfd [concrete]
// CHECK:STDOUT:   %bound_method.86a: <bound method> = bound_method %int_0.6a9, %Int.as.Copy.impl.Op.specific_fn [concrete]
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
// CHECK:STDOUT:   %Core.import_ref.d12: @Int.as.Copy.impl.%Int.as.Copy.impl.Op.type (%Int.as.Copy.impl.Op.type.24b) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Int.as.Copy.impl.%Int.as.Copy.impl.Op (constants.%Int.as.Copy.impl.Op.c95)]
// CHECK:STDOUT:   %Copy.impl_witness_table.b6a = impl_witness_table (%Core.import_ref.d12), @Int.as.Copy.impl [concrete]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/parts/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.e24: @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert.type (%Core.IntLiteral.as.ImplicitAs.impl.Convert.type.f51) = import_ref Core//prelude/parts/int, loc{{\d+_\d+}}, loaded [symbolic = @Core.IntLiteral.as.ImplicitAs.impl.%Core.IntLiteral.as.ImplicitAs.impl.Convert (constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.2a1)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.132 = impl_witness_table (%Core.import_ref.e24), @Core.IntLiteral.as.ImplicitAs.impl [concrete]
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
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4: type = splice_block %i32 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_19.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_19.1 (constants.%N.fe8)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.052) = value_binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.052) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %.Self: %type = symbolic_binding .Self [symbolic_self = constants.%.Self]
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.2: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_31.1 (%WithNontype.e02) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31: type = splice_block %WithNontype.loc6_31.2 [symbolic = %WithNontype.loc6_31.1 (constants.%WithNontype.e02)] {
// CHECK:STDOUT:       %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:       %N.ref.loc6_30: %i32 = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:       %WithNontype.loc6_31.2: type = class_type @WithNontype, @WithNontype(constants.%N.fe8) [symbolic = %WithNontype.loc6_31.1 (constants.%WithNontype.e02)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_31.1 (%WithNontype.e02) = value_binding x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param call_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_19.2: %i32) {
// CHECK:STDOUT:   %N.loc4_19.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc4_19.1 (constants.%N.fe8)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.e02
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.2: %i32) {
// CHECK:STDOUT:   %N.loc6_6.1: %i32 = symbolic_binding N, 0 [symbolic = %N.loc6_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:   %WithNontype.loc6_31.1: type = class_type @WithNontype, @WithNontype(%N.loc6_6.1) [symbolic = %WithNontype.loc6_31.1 (constants.%WithNontype.e02)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %WithNontype.loc6_31.1 [symbolic = %pattern_type (constants.%pattern_type.052)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WithNontype.loc6_31.1 [symbolic = %require_complete (constants.%require_complete.9ca)]
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.Copy.impl.Op.dfd [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.691)]
// CHECK:STDOUT:   %bound_method.loc6_50.3: <bound method> = bound_method %N.loc6_6.1, constants.%Int.as.Copy.impl.Op.specific_fn [symbolic = %bound_method.loc6_50.3 (constants.%bound_method.f2b)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%WithNontype.loc6_31.1 (%WithNontype.e02)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_50: %i32 = name_ref N, %N.loc6_6.2 [symbolic = %N.loc6_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:     %impl.elem0: %.65f = impl_witness_access constants.%Copy.impl_witness.fb7, element0 [concrete = constants.%Int.as.Copy.impl.Op.dfd]
// CHECK:STDOUT:     %bound_method.loc6_50.1: <bound method> = bound_method %N.ref.loc6_50, %impl.elem0 [symbolic = %Int.as.Copy.impl.Op.bound (constants.%Int.as.Copy.impl.Op.bound.691)]
// CHECK:STDOUT:     %specific_fn: <specific function> = specific_function %impl.elem0, @Int.as.Copy.impl.Op(constants.%int_32) [concrete = constants.%Int.as.Copy.impl.Op.specific_fn]
// CHECK:STDOUT:     %bound_method.loc6_50.2: <bound method> = bound_method %N.ref.loc6_50, %specific_fn [symbolic = %bound_method.loc6_50.3 (constants.%bound_method.f2b)]
// CHECK:STDOUT:     %Int.as.Copy.impl.Op.call: init %i32 = call %bound_method.loc6_50.2(%N.ref.loc6_50) [symbolic = %N.loc6_6.1 (constants.%N.fe8)]
// CHECK:STDOUT:     return %Int.as.Copy.impl.Op.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_13.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.322 = impl_witness_access constants.%ImplicitAs.impl_witness.bc9, element0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.e9b]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Core.IntLiteral.as.ImplicitAs.impl.Convert(constants.%int_32) [concrete = constants.%Core.IntLiteral.as.ImplicitAs.impl.Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method.9be]
// CHECK:STDOUT:   %Core.IntLiteral.as.ImplicitAs.impl.Convert.call: init %i32 = call %bound_method.loc9_31.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_31.1: %i32 = value_of_initializer %Core.IntLiteral.as.ImplicitAs.impl.Convert.call [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_31.2: %i32 = converted %int_0, %.loc9_31.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%int_0.6a9) [concrete = constants.%WithNontype.b82]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.b82 = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.b82 = class_init (), %.loc9_13.2 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %.loc9_13.4: ref %WithNontype.b82 = temporary %.loc9_13.2, %.loc9_13.3
// CHECK:STDOUT:   %.loc9_15.1: ref %WithNontype.b82 = converted %.loc9_13.1, %.loc9_13.4
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_0.6a9) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc9_15.2: %WithNontype.b82 = acquire_value %.loc9_15.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc9_15.2)
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.bound: <bound method> = bound_method %.loc9_13.4, constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fc3
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn: <specific function> = specific_function constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.fc3, @DestroyT.binding.as_type.as.Destroy.impl.Op(constants.%facet_value) [concrete = constants.%DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_13: <bound method> = bound_method %.loc9_13.4, %DestroyT.binding.as_type.as.Destroy.impl.Op.specific_fn
// CHECK:STDOUT:   %DestroyT.binding.as_type.as.Destroy.impl.Op.call: init %empty_tuple.type = call %bound_method.loc9_13(%.loc9_13.4)
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc4_19.1 => constants.%N.fe8
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.fe8) {
// CHECK:STDOUT:   %N.loc6_6.1 => constants.%N.fe8
// CHECK:STDOUT:   %WithNontype.loc6_31.1 => constants.%WithNontype.e02
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.052
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc4_19.1 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc6_6.1 => constants.%int_0.6a9
// CHECK:STDOUT:   %WithNontype.loc6_31.1 => constants.%WithNontype.b82
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b66
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT:   %Int.as.Copy.impl.Op.bound => constants.%Int.as.Copy.impl.Op.bound.986
// CHECK:STDOUT:   %bound_method.loc6_50.3 => constants.%bound_method.86a
// CHECK:STDOUT: }
// CHECK:STDOUT:
