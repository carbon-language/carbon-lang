// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C.f2e: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %pattern_type.e5e: type = pattern_type %C.f2e [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.389: <witness> = require_complete_type %C.f2e [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
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
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc7_16 (%pattern_type.e5e) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc7_16 (%pattern_type.e5e) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc7_25 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc7_25 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc7_28: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_22.2 (%C.f2e) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %C.loc7_22.1 [symbolic = %C.loc7_22.2 (constants.%C.f2e)] {
// CHECK:STDOUT:       %C.ref: %C.type = name_ref C, file.%C.decl [concrete = constants.%C.generic]
// CHECK:STDOUT:       %T.ref.loc7_21: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:       %C.loc7_22.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_22.2 (constants.%C.f2e)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%C.loc7_22.2 (%C.f2e) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.2 (%T) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.7f8 = binding_pattern p [concrete]
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
// CHECK:STDOUT:     %p: %C.131 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %D = out_param call_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.f2e
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %C.loc7_22.2: type = class_type @C, @C(%T.loc7_6.2) [symbolic = %C.loc7_22.2 (constants.%C.f2e)]
// CHECK:STDOUT:   %pattern_type.loc7_16: type = pattern_type %C.loc7_22.2 [symbolic = %pattern_type.loc7_16 (constants.%pattern_type.e5e)]
// CHECK:STDOUT:   %pattern_type.loc7_25: type = pattern_type %T.loc7_6.2 [symbolic = %pattern_type.loc7_25 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_25: <witness> = require_complete_type %T.loc7_6.2 [symbolic = %require_complete.loc7_25 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc7_17: <witness> = require_complete_type %C.loc7_22.2 [symbolic = %require_complete.loc7_17 (constants.%require_complete.389)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%C.loc7_22.2 (%C.f2e)) -> @F.%T.loc7_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_22.2 (%C.f2e) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_6.2 (%T) = call %F.specific_fn.loc7_39.1(%p.ref)
// CHECK:STDOUT:     %.loc7_43.1: @F.%T.loc7_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_43.2: @F.%T.loc7_6.2 (%T) = converted %F.call, %.loc7_43.1
// CHECK:STDOUT:     return %.loc7_43.2
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
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %C.loc7_22.2 => constants.%C.f2e
// CHECK:STDOUT:   %pattern_type.loc7_16 => constants.%pattern_type.e5e
// CHECK:STDOUT:   %pattern_type.loc7_25 => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc7_25 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc7_17 => constants.%require_complete.389
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T.loc7_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%D
// CHECK:STDOUT:   %C.loc7_22.2 => constants.%C.131
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %I.type: type = generic_class_type @I [concrete]
// CHECK:STDOUT:   %I.generic: %I.type = struct_value () [concrete]
// CHECK:STDOUT:   %I.ff1: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %pattern_type.576: type = pattern_type %I.ff1 [symbolic]
// CHECK:STDOUT:   %pattern_type.c48: type = pattern_type %C [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.ff1 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.ef1: <specific function> = specific_function %F, @F(%T) [symbolic]
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
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type (%pattern_type.576) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type (%pattern_type.576) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.c48 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.c48 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_22.2 (%I.ff1) = value_param call_param0
// CHECK:STDOUT:     %.loc7_22: type = splice_block %I.loc7_22.1 [symbolic = %I.loc7_22.2 (constants.%I.ff1)] {
// CHECK:STDOUT:       %I.ref: %I.type = name_ref I, file.%I.decl [concrete = constants.%I.generic]
// CHECK:STDOUT:       %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:       %I.loc7_22.1: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_22.2 (constants.%I.ff1)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%I.loc7_22.2 (%I.ff1) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.917 = binding_pattern p [concrete]
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
// CHECK:STDOUT:     %p: %I.ed8 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param call_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.ff1
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %I.loc7_22.2: type = class_type @I, @I(%T.loc7_6.2) [symbolic = %I.loc7_22.2 (constants.%I.ff1)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %I.loc7_22.2 [symbolic = %pattern_type (constants.%pattern_type.576)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %I.loc7_22.2 [symbolic = %require_complete (constants.%require_complete)]
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.ef1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%I.loc7_22.2 (%I.ff1)) -> %return.param: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_22.2 (%I.ff1) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %F.specific_fn.loc7_39.2 (constants.%F.specific_fn.ef1)]
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
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %I.loc7_22.2 => constants.%I.ff1
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.576
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%require_complete
// CHECK:STDOUT:   %F.specific_fn.loc7_39.2 => constants.%F.specific_fn.ef1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T.loc7_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %I.loc7_22.2 => constants.%I.ed8
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
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [concrete]
// CHECK:STDOUT:   %Outer.generic: %Outer.type = struct_value () [concrete]
// CHECK:STDOUT:   %Outer.9d6: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.eae: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.generic.137: %Inner.type.eae = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.c71: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %D: type = class_type @D [concrete]
// CHECK:STDOUT:   %require_complete.127: <witness> = require_complete_type %Outer.9d6 [symbolic]
// CHECK:STDOUT:   %pattern_type.372: type = pattern_type %Inner.c71 [symbolic]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.30b: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %pattern_type.65c: type = pattern_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.fe1: <witness> = require_complete_type %tuple.type.30b [symbolic]
// CHECK:STDOUT:   %require_complete.e7e: <witness> = require_complete_type %Inner.c71 [symbolic]
// CHECK:STDOUT:   %F.specific_fn.dd9: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %require_complete.b54: <witness> = require_complete_type %U [symbolic]
// CHECK:STDOUT:   %Outer.7c4: type = class_type @Outer, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.type.181: type = generic_class_type @Inner, @Outer(%C) [concrete]
// CHECK:STDOUT:   %Inner.generic.205: %Inner.type.181 = struct_value () [concrete]
// CHECK:STDOUT:   %Inner.d70: type = class_type @Inner, @Inner(%C, %D) [concrete]
// CHECK:STDOUT:   %pattern_type.057: type = pattern_type %Inner.d70 [concrete]
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
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [concrete = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [concrete = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:     %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     %p.patt: @F.%pattern_type.loc13_26 (%pattern_type.372) = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: @F.%pattern_type.loc13_26 (%pattern_type.372) = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @F.%pattern_type.loc13_48 (%pattern_type.65c) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @F.%pattern_type.loc13_48 (%pattern_type.65c) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.ref.loc13_52: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_55: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %.loc13_56.1: %tuple.type.24b = tuple_literal (%T.ref.loc13_52, %U.ref.loc13_55)
// CHECK:STDOUT:     %.loc13_56.2: type = converted %.loc13_56.1, constants.%tuple.type.30b [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc13_16.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_45.2 (%Inner.c71) = value_param call_param0
// CHECK:STDOUT:     %.loc13_45: type = splice_block %Inner.loc13_45.1 [symbolic = %Inner.loc13_45.2 (constants.%Inner.c71)] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %T.ref.loc13_35: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:       %Outer.loc13_36.1: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_36.2 (constants.%Outer.9d6)]
// CHECK:STDOUT:       %.loc13_37: @F.%Inner.type (%Inner.type.eae) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:       %Inner.ref: @F.%Inner.type (%Inner.type.eae) = name_ref Inner, %.loc13_37 [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:       %U.ref.loc13_44: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:       %Inner.loc13_45.1: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_45.2 (constants.%Inner.c71)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: @F.%Inner.loc13_45.2 (%Inner.c71) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.30b) = out_param call_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.30b) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type.057 = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.057 = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.9ec = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.9ec = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:     %.loc15_36.1: %tuple.type.24b = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35)
// CHECK:STDOUT:     %.loc15_36.2: type = converted %.loc15_36.1, constants.%tuple.type.e8a [concrete = constants.%tuple.type.e8a]
// CHECK:STDOUT:     %p.param: %Inner.d70 = value_param call_param0
// CHECK:STDOUT:     %.loc15_25: type = splice_block %Inner [concrete = constants.%Inner.d70] {
// CHECK:STDOUT:       %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [concrete = constants.%Outer.generic]
// CHECK:STDOUT:       %C.ref.loc15_15: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %Outer: type = class_type @Outer, @Outer(constants.%C) [concrete = constants.%Outer.7c4]
// CHECK:STDOUT:       %.loc15_17: %Inner.type.181 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [concrete = constants.%Inner.generic.205]
// CHECK:STDOUT:       %Inner.ref: %Inner.type.181 = name_ref Inner, %.loc15_17 [concrete = constants.%Inner.generic.205]
// CHECK:STDOUT:       %D.ref.loc15_24: type = name_ref D, file.%D.decl [concrete = constants.%D]
// CHECK:STDOUT:       %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [concrete = constants.%Inner.d70]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %Inner.d70 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.e8a = out_param call_param1
// CHECK:STDOUT:     %return: ref %tuple.type.e8a = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.eae)]
// CHECK:STDOUT:   %Inner.generic: @Outer.%Inner.type (%Inner.type.eae) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.eae) = class_decl @Inner [symbolic = @Outer.%Inner.generic (constants.%Inner.generic.137)] {
// CHECK:STDOUT:       %U.patt: %pattern_type.98f = symbolic_binding_pattern U, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.9d6
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.1: type, %U.loc5_15.1: type) {
// CHECK:STDOUT:   %U.loc5_15.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.c71
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type, %U.loc13_16.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc13_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:   %Outer.loc13_36.2: type = class_type @Outer, @Outer(%T.loc13_6.2) [symbolic = %Outer.loc13_36.2 (constants.%Outer.9d6)]
// CHECK:STDOUT:   %require_complete.loc13_37: <witness> = require_complete_type %Outer.loc13_36.2 [symbolic = %require_complete.loc13_37 (constants.%require_complete.127)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc13_6.2) [symbolic = %Inner.type (constants.%Inner.type.eae)]
// CHECK:STDOUT:   %Inner.generic: @F.%Inner.type (%Inner.type.eae) = struct_value () [symbolic = %Inner.generic (constants.%Inner.generic.137)]
// CHECK:STDOUT:   %Inner.loc13_45.2: type = class_type @Inner, @Inner(%T.loc13_6.2, %U.loc13_16.2) [symbolic = %Inner.loc13_45.2 (constants.%Inner.c71)]
// CHECK:STDOUT:   %pattern_type.loc13_26: type = pattern_type %Inner.loc13_45.2 [symbolic = %pattern_type.loc13_26 (constants.%pattern_type.372)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (%T.loc13_6.2, %U.loc13_16.2) [symbolic = %tuple.type (constants.%tuple.type.30b)]
// CHECK:STDOUT:   %pattern_type.loc13_48: type = pattern_type %tuple.type [symbolic = %pattern_type.loc13_48 (constants.%pattern_type.65c)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48: <witness> = require_complete_type %tuple.type [symbolic = %require_complete.loc13_48 (constants.%require_complete.fe1)]
// CHECK:STDOUT:   %require_complete.loc13_27: <witness> = require_complete_type %Inner.loc13_45.2 [symbolic = %require_complete.loc13_27 (constants.%require_complete.e7e)]
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2: <specific function> = specific_function constants.%F, @F(%T.loc13_6.2, %U.loc13_16.2) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:   %require_complete.loc13_70.1: <witness> = require_complete_type %T.loc13_6.2 [symbolic = %require_complete.loc13_70.1 (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %require_complete.loc13_70.2: <witness> = require_complete_type %U.loc13_16.2 [symbolic = %require_complete.loc13_70.2 (constants.%require_complete.b54)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%p.param: @F.%Inner.loc13_45.2 (%Inner.c71)) -> %return.param: @F.%tuple.type (%tuple.type.30b) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_45.2 (%Inner.c71) = name_ref p, %p
// CHECK:STDOUT:     %F.specific_fn.loc13_67.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %F.specific_fn.loc13_67.2 (constants.%F.specific_fn.dd9)]
// CHECK:STDOUT:     %.loc13_70.1: ref @F.%tuple.type (%tuple.type.30b) = temporary_storage
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.30b) = call %F.specific_fn.loc13_67.1(%p.ref) to %.loc13_70.1
// CHECK:STDOUT:     %.loc13_70.2: ref @F.%tuple.type (%tuple.type.30b) = temporary %.loc13_70.1, %F.call
// CHECK:STDOUT:     %tuple.elem0.loc13_70.1: ref @F.%T.loc13_6.2 (%T) = tuple_access %.loc13_70.2, element0
// CHECK:STDOUT:     %.loc13_70.3: @F.%T.loc13_6.2 (%T) = bind_value %tuple.elem0.loc13_70.1
// CHECK:STDOUT:     %tuple.elem0.loc13_70.2: ref @F.%T.loc13_6.2 (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc13_70.4: init @F.%T.loc13_6.2 (%T) = initialize_from %.loc13_70.3 to %tuple.elem0.loc13_70.2
// CHECK:STDOUT:     %tuple.elem1.loc13_70.1: ref @F.%U.loc13_16.2 (%U) = tuple_access %.loc13_70.2, element1
// CHECK:STDOUT:     %.loc13_70.5: @F.%U.loc13_16.2 (%U) = bind_value %tuple.elem1.loc13_70.1
// CHECK:STDOUT:     %tuple.elem1.loc13_70.2: ref @F.%U.loc13_16.2 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc13_70.6: init @F.%U.loc13_16.2 (%U) = initialize_from %.loc13_70.5 to %tuple.elem1.loc13_70.2
// CHECK:STDOUT:     %.loc13_70.7: init @F.%tuple.type (%tuple.type.30b) = tuple_init (%.loc13_70.4, %.loc13_70.6) to %return
// CHECK:STDOUT:     %.loc13_71: init @F.%tuple.type (%tuple.type.30b) = converted %F.call, %.loc13_70.7
// CHECK:STDOUT:     return %.loc13_71 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %Inner.d70) -> %return.param: %tuple.type.e8a {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.d70 = name_ref p, %p
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [concrete = constants.%F.specific_fn.4a7]
// CHECK:STDOUT:   %.loc15_28: ref %tuple.type.e8a = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %tuple.type.e8a = call %F.specific_fn(%p.ref) to %.loc15_28
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.eae
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_36.2 => constants.%Outer.9d6
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%require_complete.127
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.eae
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.137
// CHECK:STDOUT:   %Inner.loc13_45.2 => constants.%Inner.c71
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.372
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.30b
// CHECK:STDOUT:   %pattern_type.loc13_48 => constants.%pattern_type.65c
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%require_complete.fe1
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%require_complete.e7e
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.dd9
// CHECK:STDOUT:   %require_complete.loc13_70.1 => constants.%require_complete.4ae
// CHECK:STDOUT:   %require_complete.loc13_70.2 => constants.%require_complete.b54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@F.%T.loc13_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@F.%T.loc13_6.2, @F.%U.loc13_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc13_6.2, %U.loc13_16.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.181
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.205
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%C, constants.%D) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%D
// CHECK:STDOUT:   %Outer.loc13_36.2 => constants.%Outer.7c4
// CHECK:STDOUT:   %require_complete.loc13_37 => constants.%complete_type.357
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.181
// CHECK:STDOUT:   %Inner.generic => constants.%Inner.generic.205
// CHECK:STDOUT:   %Inner.loc13_45.2 => constants.%Inner.d70
// CHECK:STDOUT:   %pattern_type.loc13_26 => constants.%pattern_type.057
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.e8a
// CHECK:STDOUT:   %pattern_type.loc13_48 => constants.%pattern_type.9ec
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc13_48 => constants.%complete_type.53b
// CHECK:STDOUT:   %require_complete.loc13_27 => constants.%complete_type.357
// CHECK:STDOUT:   %F.specific_fn.loc13_67.2 => constants.%F.specific_fn.4a7
// CHECK:STDOUT:   %require_complete.loc13_70.1 => constants.%complete_type.357
// CHECK:STDOUT:   %require_complete.loc13_70.2 => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontype.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %N.51e: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.7ce: type = pattern_type %i32 [concrete]
// CHECK:STDOUT:   %WithNontype.type: type = generic_class_type @WithNontype [concrete]
// CHECK:STDOUT:   %WithNontype.generic: %WithNontype.type = struct_value () [concrete]
// CHECK:STDOUT:   %WithNontype.8a6: type = class_type @WithNontype, @WithNontype(%N.51e) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.48f: type = pattern_type %WithNontype.8a6 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %require_complete.9c5: <witness> = require_complete_type %WithNontype.8a6 [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_0.5c6: Core.IntLiteral = int_value 0 [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.cc7: type = generic_interface_type @ImplicitAs [concrete]
// CHECK:STDOUT:   %ImplicitAs.generic: %ImplicitAs.type.cc7 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.type.205: type = facet_type <@ImplicitAs, @ImplicitAs(%i32)> [concrete]
// CHECK:STDOUT:   %Convert.type.1b6: type = fn_type @Convert.1, @ImplicitAs(%i32) [concrete]
// CHECK:STDOUT:   %To.c80: Core.IntLiteral = bind_symbolic_name To, 0 [symbolic]
// CHECK:STDOUT:   %Convert.type.0f9: type = fn_type @Convert.2, @impl.4f9(%To.c80) [symbolic]
// CHECK:STDOUT:   %Convert.f06: %Convert.type.0f9 = struct_value () [symbolic]
// CHECK:STDOUT:   %ImplicitAs.impl_witness.c75: <witness> = impl_witness imports.%ImplicitAs.impl_witness_table.a2f, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.type.035: type = fn_type @Convert.2, @impl.4f9(%int_32) [concrete]
// CHECK:STDOUT:   %Convert.956: %Convert.type.035 = struct_value () [concrete]
// CHECK:STDOUT:   %ImplicitAs.facet: %ImplicitAs.type.205 = facet_value Core.IntLiteral, (%ImplicitAs.impl_witness.c75) [concrete]
// CHECK:STDOUT:   %.9c3: type = fn_type_with_self_type %Convert.type.1b6, %ImplicitAs.facet [concrete]
// CHECK:STDOUT:   %Convert.bound: <bound method> = bound_method %int_0.5c6, %Convert.956 [concrete]
// CHECK:STDOUT:   %Convert.specific_fn: <specific function> = specific_function %Convert.956, @Convert.2(%int_32) [concrete]
// CHECK:STDOUT:   %bound_method: <bound method> = bound_method %int_0.5c6, %Convert.specific_fn [concrete]
// CHECK:STDOUT:   %int_0.6a9: %i32 = int_value 0 [concrete]
// CHECK:STDOUT:   %WithNontype.b82: type = class_type @WithNontype, @WithNontype(%int_0.6a9) [concrete]
// CHECK:STDOUT:   %WithNontype.val: %WithNontype.b82 = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.b66: type = pattern_type %WithNontype.b82 [concrete]
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F, @F(%int_0.6a9) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     .ImplicitAs = %Core.ImplicitAs
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT:   %Core.ImplicitAs: %ImplicitAs.type.cc7 = import_ref Core//prelude/operators/as, ImplicitAs, loaded [concrete = constants.%ImplicitAs.generic]
// CHECK:STDOUT:   %Core.import_ref.a5b: @impl.4f9.%Convert.type (%Convert.type.0f9) = import_ref Core//prelude/types/int, loc19_39, loaded [symbolic = @impl.4f9.%Convert (constants.%Convert.f06)]
// CHECK:STDOUT:   %ImplicitAs.impl_witness_table.a2f = impl_witness_table (%Core.import_ref.a5b), @impl.4f9 [concrete]
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
// CHECK:STDOUT:       %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc4_19.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N.51e)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %N.patt: %pattern_type.7ce = symbolic_binding_pattern N, 0 [concrete]
// CHECK:STDOUT:     %x.patt: @F.%pattern_type (%pattern_type.48f) = binding_pattern x [concrete]
// CHECK:STDOUT:     %x.param_patt: @F.%pattern_type (%pattern_type.48f) = value_param_pattern %x.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.7ce = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.7ce = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int_32.loc6_37: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:     %i32.loc6_37: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     %.loc6_10: type = splice_block %i32.loc6_10 [concrete = constants.%i32] {
// CHECK:STDOUT:       %int_32.loc6_10: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:       %i32.loc6_10: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %N.loc6_6.1: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_31.2 (%WithNontype.8a6) = value_param call_param0
// CHECK:STDOUT:     %.loc6_31: type = splice_block %WithNontype.loc6_31.1 [symbolic = %WithNontype.loc6_31.2 (constants.%WithNontype.8a6)] {
// CHECK:STDOUT:       %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:       %N.ref.loc6_30: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:       %WithNontype.loc6_31.1: type = class_type @WithNontype, @WithNontype(constants.%N.51e) [symbolic = %WithNontype.loc6_31.2 (constants.%WithNontype.8a6)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_31.2 (%WithNontype.8a6) = bind_name x, %x.param
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
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_19.1: %i32) {
// CHECK:STDOUT:   %N.loc4_19.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N.51e)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.8a6
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: %i32) {
// CHECK:STDOUT:   %N.loc6_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:   %WithNontype.loc6_31.2: type = class_type @WithNontype, @WithNontype(%N.loc6_6.2) [symbolic = %WithNontype.loc6_31.2 (constants.%WithNontype.8a6)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %WithNontype.loc6_31.2 [symbolic = %pattern_type (constants.%pattern_type.48f)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %WithNontype.loc6_31.2 [symbolic = %require_complete (constants.%require_complete.9c5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%x.param: @F.%WithNontype.loc6_31.2 (%WithNontype.8a6)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_50: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.51e)]
// CHECK:STDOUT:     return %N.ref.loc6_50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [concrete = constants.%F]
// CHECK:STDOUT:   %.loc9_13.1: %empty_struct_type = struct_literal ()
// CHECK:STDOUT:   %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [concrete = constants.%WithNontype.generic]
// CHECK:STDOUT:   %int_0: Core.IntLiteral = int_value 0 [concrete = constants.%int_0.5c6]
// CHECK:STDOUT:   %impl.elem0: %.9c3 = impl_witness_access constants.%ImplicitAs.impl_witness.c75, element0 [concrete = constants.%Convert.956]
// CHECK:STDOUT:   %bound_method.loc9_31.1: <bound method> = bound_method %int_0, %impl.elem0 [concrete = constants.%Convert.bound]
// CHECK:STDOUT:   %specific_fn: <specific function> = specific_function %impl.elem0, @Convert.2(constants.%int_32) [concrete = constants.%Convert.specific_fn]
// CHECK:STDOUT:   %bound_method.loc9_31.2: <bound method> = bound_method %int_0, %specific_fn [concrete = constants.%bound_method]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %bound_method.loc9_31.2(%int_0) [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_31.1: %i32 = value_of_initializer %int.convert_checked [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %.loc9_31.2: %i32 = converted %int_0, %.loc9_31.1 [concrete = constants.%int_0.6a9]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%int_0.6a9) [concrete = constants.%WithNontype.b82]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.b82 = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.b82 = class_init (), %.loc9_13.2 [concrete = constants.%WithNontype.val]
// CHECK:STDOUT:   %.loc9_13.4: ref %WithNontype.b82 = temporary %.loc9_13.2, %.loc9_13.3
// CHECK:STDOUT:   %.loc9_15.1: ref %WithNontype.b82 = converted %.loc9_13.1, %.loc9_13.4
// CHECK:STDOUT:   %F.specific_fn: <specific function> = specific_function %F.ref, @F(constants.%int_0.6a9) [concrete = constants.%F.specific_fn]
// CHECK:STDOUT:   %.loc9_15.2: %WithNontype.b82 = bind_value %.loc9_15.1
// CHECK:STDOUT:   %F.call: init %i32 = call %F.specific_fn(%.loc9_15.2)
// CHECK:STDOUT:   %.loc9_33.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_33.2: %i32 = converted %F.call, %.loc9_33.1
// CHECK:STDOUT:   return %.loc9_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N.51e
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.51e) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N.51e
// CHECK:STDOUT:   %WithNontype.loc6_31.2 => constants.%WithNontype.8a6
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.48f
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(@F.%N.loc6_6.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%int_0.6a9
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%int_0.6a9) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%int_0.6a9
// CHECK:STDOUT:   %WithNontype.loc6_31.2 => constants.%WithNontype.b82
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.b66
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete => constants.%complete_type.357
// CHECK:STDOUT: }
// CHECK:STDOUT:
