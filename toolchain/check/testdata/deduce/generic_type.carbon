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
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%D) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %C.decl: %C.type = class_decl @C [template = constants.%C.1] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_6.1, runtime_param<invalid> [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%C.loc7_20.2 (%C.2) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%C.loc7_20.2 (%C.2) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc7_6.2 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc7_6.2 (%T) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %T.ref.loc7_21: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc7_20.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_20.2 (constants.%C.2)]
// CHECK:STDOUT:     %T.ref.loc7_28: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_20.2 (%C.2) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%C.loc7_20.2 (%C.2) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%T.loc7_6.2 (%T) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.2 (%T) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %C.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %C.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %D = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %D = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %D.ref.loc9_11: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%D) [template = constants.%C.3]
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %p.param: %C.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %C.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %D = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %D = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %C.loc7_20.2: type = class_type @C, @C(%T.loc7_6.2) [symbolic = %C.loc7_20.2 (constants.%C.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2) [symbolic = %.loc7_39.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%C.loc7_20.2 (%C.2)) -> @F.%T.loc7_6.2 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%C.loc7_20.2 (%C.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc7_39.2 (constants.%.4)]
// CHECK:STDOUT:     %F.call: init @F.%T.loc7_6.2 (%T) = call %.loc7_39.1(%p.ref)
// CHECK:STDOUT:     %.loc7_43.1: @F.%T.loc7_6.2 (%T) = value_of_initializer %F.call
// CHECK:STDOUT:     %.loc7_43.2: @F.%T.loc7_6.2 (%T) = converted %F.call, %.loc7_43.1
// CHECK:STDOUT:     return %.loc7_43.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %C.3) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc10: <specific function> = specific_function %F.ref, @F(constants.%D) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_15.2: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %.loc10(%p.ref) to %.loc9_15.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%D
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%D
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_class_type @I [template]
// CHECK:STDOUT:   %I.1: %I.type = struct_value () [template]
// CHECK:STDOUT:   %I.2: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: <specific function> = specific_function %F, @F(%T) [symbolic]
// CHECK:STDOUT:   %I.3: type = class_type @I, @I(%C) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%C) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .I = %I.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %I.decl: %I.type = class_decl @I [template = constants.%I.1] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_9.1, runtime_param<invalid> [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc7_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc7_6.1, runtime_param<invalid> [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %p.patt: @F.%I.loc7_20.2 (%I.2) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%I.loc7_20.2 (%I.2) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %I.loc7_20.1: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_20.2 (constants.%I.2)]
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_20.2 (%I.2) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%I.loc7_20.2 (%I.2) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %I.3 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %I.3 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %C = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %C = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I.1]
// CHECK:STDOUT:     %C.ref.loc9_11: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I: type = class_type @I, @I(constants.%C) [template = constants.%I.3]
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %p.param: %I.3 = value_param runtime_param0
// CHECK:STDOUT:     %p: %I.3 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %C = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %C = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc7_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc7_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %I.loc7_20.2: type = class_type @I, @I(%T.loc7_6.2) [symbolic = %I.loc7_20.2 (constants.%I.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2: <specific function> = specific_function constants.%F, @F(%T.loc7_6.2) [symbolic = %.loc7_39.2 (constants.%.4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type](%p.param_patt: @F.%I.loc7_20.2 (%I.2)) -> %return: %C {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%I.loc7_20.2 (%I.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc7_39.1: <specific function> = specific_function %F.ref, @F(constants.%T) [symbolic = %.loc7_39.2 (constants.%.4)]
// CHECK:STDOUT:     %.loc7_25.2: ref %C = splice_block %return {}
// CHECK:STDOUT:     %F.call: init %C = call %.loc7_39.1(%p.ref) to %.loc7_25.2
// CHECK:STDOUT:     return %F.call to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %I.3) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc10: <specific function> = specific_function %F.ref, @F(constants.%C) [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_15.2: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %.loc10(%p.ref) to %.loc9_15.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %I.loc7_20.2 => constants.%I.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2 => constants.%.4
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %I.loc7_20.2 => constants.%I.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %I.loc7_20.2 => constants.%I.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc7_39.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %Outer.1: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.1: %Inner.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.1: type = struct_type {} [template]
// CHECK:STDOUT:   %.2: <witness> = complete_type_witness %.1 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %tuple.type.1: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %tuple.type.2: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.5: <specific function> = specific_function %F, @F(%T, %U) [symbolic]
// CHECK:STDOUT:   %Outer.3: type = class_type @Outer, @Outer(%C) [template]
// CHECK:STDOUT:   %Inner.type.2: type = generic_class_type @Inner, @Outer(%C) [template]
// CHECK:STDOUT:   %Inner.3: %Inner.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Inner.4: type = class_type @Inner, @Inner(%C, %D) [template]
// CHECK:STDOUT:   %tuple.type.3: type = tuple_type (%C, %D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.7: <specific function> = specific_function %F, @F(%C, %D) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Outer = %Outer.decl
// CHECK:STDOUT:     .C = %C.decl
// CHECK:STDOUT:     .D = %D.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Outer.decl: %Outer.type = class_decl @Outer [template = constants.%Outer.1] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc4_13.1, runtime_param<invalid> [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt.loc13_6.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %T.param_patt: type = value_param_pattern %T.patt.loc13_6.1, runtime_param<invalid> [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:     %U.patt.loc13_16.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %U.param_patt: type = value_param_pattern %U.patt.loc13_16.1, runtime_param<invalid> [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:     %p.patt: @F.%Inner.loc13_43.2 (%Inner.2) = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: @F.%Inner.loc13_43.2 (%Inner.2) = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: @F.%tuple.type (%tuple.type.2) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%tuple.type (%tuple.type.2) = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:     %T.ref.loc13_35: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %Outer.loc13_34.1: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_34.2 (constants.%Outer.2)]
// CHECK:STDOUT:     %.loc13_37: @F.%Inner.type (%Inner.type.1) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:     %Inner.ref: @F.%Inner.type (%Inner.type.1) = name_ref Inner, %.loc13_37 [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:     %U.ref.loc13_44: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %Inner.loc13_43.1: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_43.2 (constants.%Inner.2)]
// CHECK:STDOUT:     %T.ref.loc13_52: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_55: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %.loc13_56.1: %tuple.type.1 = tuple_literal (%T.ref.loc13_52, %U.ref.loc13_55)
// CHECK:STDOUT:     %.loc13_56.2: type = converted %.loc13_56.1, constants.%tuple.type.2 [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:     %T.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:     %U.loc13_16.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_43.2 (%Inner.2) = value_param runtime_param0
// CHECK:STDOUT:     %p: @F.%Inner.loc13_43.2 (%Inner.2) = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref @F.%tuple.type (%tuple.type.2) = out_param runtime_param1
// CHECK:STDOUT:     %return: ref @F.%tuple.type (%tuple.type.2) = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %Inner.4 = binding_pattern p
// CHECK:STDOUT:     %p.param_patt: %Inner.4 = value_param_pattern %p.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %tuple.type.3 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %tuple.type.3 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:     %C.ref.loc15_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%C) [template = constants.%Outer.3]
// CHECK:STDOUT:     %.loc15_17: %Inner.type.2 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [template = constants.%Inner.3]
// CHECK:STDOUT:     %Inner.ref: %Inner.type.2 = name_ref Inner, %.loc15_17 [template = constants.%Inner.3]
// CHECK:STDOUT:     %D.ref.loc15_24: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [template = constants.%Inner.4]
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc15_36.1: %tuple.type.1 = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35)
// CHECK:STDOUT:     %.loc15_36.2: type = converted %.loc15_36.1, constants.%tuple.type.3 [template = constants.%tuple.type.3]
// CHECK:STDOUT:     %p.param: %Inner.4 = value_param runtime_param0
// CHECK:STDOUT:     %p: %Inner.4 = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %tuple.type.3 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %tuple.type.3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner: @Outer.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner (constants.%Inner.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.1) = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.1)] {
// CHECK:STDOUT:       %U.patt.loc5_15.1: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:       %U.param_patt: type = value_param_pattern %U.patt.loc5_15.1, runtime_param<invalid> [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = value_param runtime_param<invalid>
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2
// CHECK:STDOUT:     .Inner = %Inner.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Inner(@Outer.%T.loc4_13.1: type, %U.loc5_15.1: type) {
// CHECK:STDOUT:   %U.loc5_15.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc5_15.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc5_15.2 (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc5: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.1 [template = constants.%.2]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type, %U.loc13_16.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc13_6.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc13_6.2 (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc13_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt.loc13_16.2: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt.loc13_16.2 (constants.%U.patt)]
// CHECK:STDOUT:   %Outer.loc13_34.2: type = class_type @Outer, @Outer(%T.loc13_6.2) [symbolic = %Outer.loc13_34.2 (constants.%Outer.2)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc13_6.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner.loc13_37: @F.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:   %Inner.loc13_43.2: type = class_type @Inner, @Inner(%T.loc13_6.2, %U.loc13_16.2) [symbolic = %Inner.loc13_43.2 (constants.%Inner.2)]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (@F.%T.loc13_6.2 (%T), @F.%U.loc13_16.2 (%U)) [symbolic = %tuple.type (constants.%tuple.type.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_67.2: <specific function> = specific_function constants.%F, @F(%T.loc13_6.2, %U.loc13_16.2) [symbolic = %.loc13_67.2 (constants.%.5)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.param_patt: type, %U.param_patt: type](%p.param_patt: @F.%Inner.loc13_43.2 (%Inner.2)) -> %return: @F.%tuple.type (%tuple.type.2) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:     %p.ref: @F.%Inner.loc13_43.2 (%Inner.2) = name_ref p, %p
// CHECK:STDOUT:     %.loc13_67.1: <specific function> = specific_function %F.ref, @F(constants.%T, constants.%U) [symbolic = %.loc13_67.2 (constants.%.5)]
// CHECK:STDOUT:     %.loc13_68.1: ref @F.%tuple.type (%tuple.type.2) = temporary_storage
// CHECK:STDOUT:     %F.call: init @F.%tuple.type (%tuple.type.2) = call %.loc13_67.1(%p.ref) to %.loc13_68.1
// CHECK:STDOUT:     %.loc13_68.2: ref @F.%tuple.type (%tuple.type.2) = temporary %.loc13_68.1, %F.call
// CHECK:STDOUT:     %.loc13_68.3: ref @F.%T.loc13_6.2 (%T) = tuple_access %.loc13_68.2, element0
// CHECK:STDOUT:     %.loc13_68.4: @F.%T.loc13_6.2 (%T) = bind_value %.loc13_68.3
// CHECK:STDOUT:     %.loc13_68.5: ref @F.%T.loc13_6.2 (%T) = tuple_access %return, element0
// CHECK:STDOUT:     %.loc13_68.6: init @F.%T.loc13_6.2 (%T) = initialize_from %.loc13_68.4 to %.loc13_68.5
// CHECK:STDOUT:     %.loc13_68.7: ref @F.%U.loc13_16.2 (%U) = tuple_access %.loc13_68.2, element1
// CHECK:STDOUT:     %.loc13_68.8: @F.%U.loc13_16.2 (%U) = bind_value %.loc13_68.7
// CHECK:STDOUT:     %.loc13_68.9: ref @F.%U.loc13_16.2 (%U) = tuple_access %return, element1
// CHECK:STDOUT:     %.loc13_68.10: init @F.%U.loc13_16.2 (%U) = initialize_from %.loc13_68.8 to %.loc13_68.9
// CHECK:STDOUT:     %.loc13_68.11: init @F.%tuple.type (%tuple.type.2) = tuple_init (%.loc13_68.6, %.loc13_68.10) to %return
// CHECK:STDOUT:     %.loc13_71: init @F.%tuple.type (%tuple.type.2) = converted %F.call, %.loc13_68.11
// CHECK:STDOUT:     return %.loc13_71 to %return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param_patt: %Inner.4) -> %return: %tuple.type.3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc16: <specific function> = specific_function %F.ref, @F(constants.%C, constants.%D) [template = constants.%.7]
// CHECK:STDOUT:   %.loc15_28.2: ref %tuple.type.3 = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %tuple.type.3 = call %.loc16(%p.ref) to %.loc15_28.2
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner => constants.%Inner.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%U
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@F.%T.loc13_6.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@F.%T.loc13_6.2, @F.%U.loc13_16.2) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_34.2 => constants.%Outer.2
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner.loc13_37 => constants.%Inner.1
// CHECK:STDOUT:   %Inner.loc13_43.2 => constants.%Inner.2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_67.2 => constants.%.5
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(%T.loc13_6.2, %U.loc13_16.2) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_34.2 => constants.%Outer.2
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner.loc13_37 => constants.%Inner.1
// CHECK:STDOUT:   %Inner.loc13_43.2 => constants.%Inner.2
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner => constants.%Inner.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%C, constants.%D) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc5_15.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C, constants.%D) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %T.patt.loc13_6.2 => constants.%C
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%D
// CHECK:STDOUT:   %U.patt.loc13_16.2 => constants.%D
// CHECK:STDOUT:   %Outer.loc13_34.2 => constants.%Outer.3
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner.loc13_37 => constants.%Inner.3
// CHECK:STDOUT:   %Inner.loc13_43.2 => constants.%Inner.4
// CHECK:STDOUT:   %tuple.type => constants.%tuple.type.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %.loc13_67.2 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontype.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.1: Core.IntLiteral = int_value 32 [template]
// CHECK:STDOUT:   %Int.type: type = fn_type @Int [template]
// CHECK:STDOUT:   %Int: %Int.type = struct_value () [template]
// CHECK:STDOUT:   %i32: type = int_type signed, %.1 [template]
// CHECK:STDOUT:   %N.1: %i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %N.patt.1: %i32 = symbolic_binding_pattern N, 0 [symbolic]
// CHECK:STDOUT:   %WithNontype.type: type = generic_class_type @WithNontype [template]
// CHECK:STDOUT:   %WithNontype.1: %WithNontype.type = struct_value () [template]
// CHECK:STDOUT:   %WithNontype.2: type = class_type @WithNontype, @WithNontype(%N.1) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: Core.IntLiteral = int_value 0 [template]
// CHECK:STDOUT:   %Convert.type.2: type = fn_type @Convert.1, @ImplicitAs(%i32) [template]
// CHECK:STDOUT:   %Convert.type.14: type = fn_type @Convert.2, @impl.1(%.1) [template]
// CHECK:STDOUT:   %Convert.14: %Convert.type.14 = struct_value () [template]
// CHECK:STDOUT:   %.29: <witness> = interface_witness (%Convert.14) [template]
// CHECK:STDOUT:   %.30: <bound method> = bound_method %.5, %Convert.14 [template]
// CHECK:STDOUT:   %.31: <specific function> = specific_function %.30, @Convert.2(%.1) [template]
// CHECK:STDOUT:   %.32: %i32 = int_value 0 [template]
// CHECK:STDOUT:   %WithNontype.3: type = class_type @WithNontype, @WithNontype(%.32) [template]
// CHECK:STDOUT:   %struct: %WithNontype.3 = struct_value () [template]
// CHECK:STDOUT:   %.33: <specific function> = specific_function %F, @F(%.32) [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int = %import_ref.1
// CHECK:STDOUT:     .ImplicitAs = %import_ref.2
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [template] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .WithNontype = %WithNontype.decl
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %WithNontype.decl: %WithNontype.type = class_decl @WithNontype [template = constants.%WithNontype.1] {
// CHECK:STDOUT:     %N.patt.loc4_19.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_19.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc4_19.1, runtime_param<invalid> [symbolic = %N.patt.loc4_19.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc4_23.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc4_23.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_23.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc4_23.3: type = converted %int.make_type_signed, %.loc4_23.2 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_19.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_19.2 (constants.%N.1)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt.loc6_6.1: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %N.param_patt: %i32 = value_param_pattern %N.patt.loc6_6.1, runtime_param<invalid> [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:     %x.patt: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = binding_pattern x
// CHECK:STDOUT:     %x.param_patt: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = value_param_pattern %x.patt, runtime_param0
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc6_10.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_10: init type = call constants.%Int(%.loc6_10.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_10.2: type = value_of_initializer %int.make_type_signed.loc6_10 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_10.3: type = converted %int.make_type_signed.loc6_10, %.loc6_10.2 [template = constants.%i32]
// CHECK:STDOUT:     %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [template = constants.%WithNontype.1]
// CHECK:STDOUT:     %N.ref.loc6_30: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %WithNontype.loc6_29.1: type = class_type @WithNontype, @WithNontype(constants.%N.1) [symbolic = %WithNontype.loc6_29.2 (constants.%WithNontype.2)]
// CHECK:STDOUT:     %.loc6_37.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed.loc6_37: init type = call constants.%Int(%.loc6_37.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_37.2: type = value_of_initializer %int.make_type_signed.loc6_37 [template = constants.%i32]
// CHECK:STDOUT:     %.loc6_37.3: type = converted %int.make_type_signed.loc6_37, %.loc6_37.2 [template = constants.%i32]
// CHECK:STDOUT:     %N.param: %i32 = value_param runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_6.1: %i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_6.2 (constants.%N.1)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = value_param runtime_param0
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = bind_name x, %x.param
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param1
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %return.patt: %i32 = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: %i32 = out_param_pattern %return.patt, runtime_param0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc8_11.1: Core.IntLiteral = int_value 32 [template = constants.%.1]
// CHECK:STDOUT:     %int.make_type_signed: init type = call constants.%Int(%.loc8_11.1) [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_11.2: type = value_of_initializer %int.make_type_signed [template = constants.%i32]
// CHECK:STDOUT:     %.loc8_11.3: type = converted %int.make_type_signed, %.loc8_11.2 [template = constants.%i32]
// CHECK:STDOUT:     %return.param: ref %i32 = out_param runtime_param0
// CHECK:STDOUT:     %return: ref %i32 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_19.1: %i32) {
// CHECK:STDOUT:   %N.loc4_19.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc4_19.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc4_19.2 (constants.%N.patt.1)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4_29: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%WithNontype.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: %i32) {
// CHECK:STDOUT:   %N.loc6_6.2: %i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N.1)]
// CHECK:STDOUT:   %N.patt.loc6_6.2: %i32 = symbolic_binding_pattern N, 0 [symbolic = %N.patt.loc6_6.2 (constants.%N.patt.1)]
// CHECK:STDOUT:   %WithNontype.loc6_29.2: type = class_type @WithNontype, @WithNontype(%N.loc6_6.2) [symbolic = %WithNontype.loc6_29.2 (constants.%WithNontype.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.param_patt: %i32](%x.param_patt: @F.%WithNontype.loc6_29.2 (%WithNontype.2)) -> %i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_50: %i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N.1)]
// CHECK:STDOUT:     return %N.ref.loc6_50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> %i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc9_13.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [template = constants.%WithNontype.1]
// CHECK:STDOUT:   %.loc9_30: Core.IntLiteral = int_value 0 [template = constants.%.5]
// CHECK:STDOUT:   %.loc9_29.1: %Convert.type.2 = interface_witness_access constants.%.29, element0 [template = constants.%Convert.14]
// CHECK:STDOUT:   %.loc9_29.2: <bound method> = bound_method %.loc9_30, %.loc9_29.1 [template = constants.%.30]
// CHECK:STDOUT:   %.loc9_29.3: <specific function> = specific_function %.loc9_29.2, @Convert.2(constants.%.1) [template = constants.%.31]
// CHECK:STDOUT:   %int.convert_checked: init %i32 = call %.loc9_29.3(%.loc9_30) [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_29.4: %i32 = value_of_initializer %int.convert_checked [template = constants.%.32]
// CHECK:STDOUT:   %.loc9_29.5: %i32 = converted %.loc9_30, %.loc9_29.4 [template = constants.%.32]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%.32) [template = constants.%WithNontype.3]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.3 = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.3 = class_init (), %.loc9_13.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_13.4: ref %WithNontype.3 = temporary %.loc9_13.2, %.loc9_13.3
// CHECK:STDOUT:   %.loc9_15.1: ref %WithNontype.3 = converted %.loc9_13.1, %.loc9_13.4
// CHECK:STDOUT:   %.loc9_10: <specific function> = specific_function %F.ref, @F(constants.%.32) [template = constants.%.33]
// CHECK:STDOUT:   %.loc9_15.2: %WithNontype.3 = bind_value %.loc9_15.1
// CHECK:STDOUT:   %F.call: init %i32 = call %.loc9_10(%.loc9_15.2)
// CHECK:STDOUT:   %.loc9_33.1: %i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_33.2: %i32 = converted %F.call, %.loc9_33.1
// CHECK:STDOUT:   return %.loc9_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N.1) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_19.2 => constants.%N.1
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(@F.%N.loc6_6.2) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc4_19.2 => constants.%N.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N.1) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N.1
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%N.1
// CHECK:STDOUT:   %WithNontype.loc6_29.2 => constants.%WithNontype.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%.32) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%.32
// CHECK:STDOUT:   %N.patt.loc4_19.2 => constants.%.32
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.32) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%.32
// CHECK:STDOUT:   %N.patt.loc6_6.2 => constants.%.32
// CHECK:STDOUT:   %WithNontype.loc6_29.2 => constants.%WithNontype.3
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
