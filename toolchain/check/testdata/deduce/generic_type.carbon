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

fn F[T:! type](p: C(T)) -> T;

fn G(p: C(D)) -> D {
  return F(p);
}

// --- interface.carbon

library "[[@TEST_NAME]]";

class I(T:! type) {}
class C {}

fn F[T:! type](p: I(T)) -> C;

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
fn F[T:! type, U:! type](p: Outer(T).Inner(U)) -> (T, U);

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
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %C.1: %C.type = struct_value () [template]
// CHECK:STDOUT:   %C.2: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %C.3: type = class_type @C, @C(%D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %p.patt: @F.%C.loc7_20.2 (%C.2) = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %T.ref.loc7_21: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %C.loc7_20.1: type = class_type @C, @C(constants.%T) [symbolic = %C.loc7_20.2 (constants.%C.2)]
// CHECK:STDOUT:     %p.param: @F.%C.loc7_20.2 (%C.2) = param p, runtime_param0
// CHECK:STDOUT:     %p: @F.%C.loc7_20.2 (%C.2) = bind_name p, %p.param
// CHECK:STDOUT:     %T.ref.loc7_28: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %return: ref @F.%T.loc7_6.2 (%T) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %C.3 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref: %C.type = name_ref C, file.%C.decl [template = constants.%C.1]
// CHECK:STDOUT:     %D.ref.loc9_11: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %C: type = class_type @C, @C(constants.%D) [template = constants.%C.3]
// CHECK:STDOUT:     %p.param: %C.3 = param p, runtime_param0
// CHECK:STDOUT:     %p: %C.3 = bind_name p, %p.param
// CHECK:STDOUT:     %D.ref.loc9_18: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %return: ref %D = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %C.loc7_20.2: type = class_type @C, @C(%T.loc7_6.2) [symbolic = %C.loc7_20.2 (constants.%C.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.loc7_6.1: type](%p: @F.%C.loc7_20.2 (%C.2)) -> @F.%T.loc7_6.2 (%T);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: %C.3) -> %return: %D {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %C.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc9: ref %D = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %D = call %F.ref(%p.ref) to %.loc9
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(@F.%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%D) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%D
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%D) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%D
// CHECK:STDOUT:   %C.loc7_20.2 => constants.%C.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- interface.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %I.type: type = generic_class_type @I [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %I.1: %I.type = struct_value () [template]
// CHECK:STDOUT:   %I.2: type = class_type @I, @I(%T) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %I.3: type = class_type @I, @I(%C) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %p.patt: @F.%I.loc7_20.2 (%I.2) = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc7_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I.1]
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc7_6.1 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:     %I.loc7_20.1: type = class_type @I, @I(constants.%T) [symbolic = %I.loc7_20.2 (constants.%I.2)]
// CHECK:STDOUT:     %p.param: @F.%I.loc7_20.2 (%I.2) = param p, runtime_param0
// CHECK:STDOUT:     %p: @F.%I.loc7_20.2 (%I.2) = bind_name p, %p.param
// CHECK:STDOUT:     %C.ref: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %I.3 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %I.ref: %I.type = name_ref I, file.%I.decl [template = constants.%I.1]
// CHECK:STDOUT:     %C.ref.loc9_11: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %I: type = class_type @I, @I(constants.%C) [template = constants.%I.3]
// CHECK:STDOUT:     %p.param: %I.3 = param p, runtime_param0
// CHECK:STDOUT:     %p: %I.3 = bind_name p, %p.param
// CHECK:STDOUT:     %C.ref.loc9_18: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %return: ref %C = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @I(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %.loc4: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%I.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc5: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc7_6.1: type) {
// CHECK:STDOUT:   %T.loc7_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc7_6.2 (constants.%T)]
// CHECK:STDOUT:   %I.loc7_20.2: type = class_type @I, @I(%T.loc7_6.2) [symbolic = %I.loc7_20.2 (constants.%I.2)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.loc7_6.1: type](%p: @F.%I.loc7_20.2 (%I.2)) -> %C;
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: %I.3) -> %return: %C {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %I.3 = name_ref p, %p
// CHECK:STDOUT:   %.loc9: ref %C = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %C = call %F.ref(%p.ref) to %.loc9
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(@F.%T.loc7_6.2) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%T
// CHECK:STDOUT:   %I.loc7_20.2 => constants.%I.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @I(constants.%C) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%C) {
// CHECK:STDOUT:   %T.loc7_6.2 => constants.%C
// CHECK:STDOUT:   %I.loc7_20.2 => constants.%I.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %Outer.type: type = generic_class_type @Outer [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Outer.1: %Outer.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.2: type = class_type @Outer, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %Inner.type.1: type = generic_class_type @Inner, @Outer(%T) [symbolic]
// CHECK:STDOUT:   %Inner.1: %Inner.type.1 = struct_value () [symbolic]
// CHECK:STDOUT:   %Inner.2: type = class_type @Inner, @Inner(%T, %U) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %C: type = class_type @C [template]
// CHECK:STDOUT:   %D: type = class_type @D [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %.5: type = tuple_type (type, type) [template]
// CHECK:STDOUT:   %.6: type = tuple_type (%T, %U) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %Outer.3: type = class_type @Outer, @Outer(%C) [template]
// CHECK:STDOUT:   %Inner.type.2: type = generic_class_type @Inner, @Outer(%C) [template]
// CHECK:STDOUT:   %Inner.3: %Inner.type.2 = struct_value () [template]
// CHECK:STDOUT:   %Inner.4: type = class_type @Inner, @Inner(%C, %D) [template]
// CHECK:STDOUT:   %.7: type = tuple_type (%C, %D) [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.8: type = tuple_type (%.4, %.4) [template]
// CHECK:STDOUT:   %.9: type = ptr_type %.8 [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
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
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %C.decl: type = class_decl @C [template = constants.%C] {} {}
// CHECK:STDOUT:   %D.decl: type = class_decl @D [template = constants.%D] {} {}
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %T.patt: type = symbolic_binding_pattern T, 0
// CHECK:STDOUT:     %U.patt: type = symbolic_binding_pattern U, 1
// CHECK:STDOUT:     %p.patt: @F.%Inner.loc13_43.2 (%Inner.2) = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.param: type = param T, runtime_param<invalid>
// CHECK:STDOUT:     %T.loc13_6.1: type = bind_symbolic_name T, 0, %T.param [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:     %U.loc13_16.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:     %T.ref.loc13_35: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %Outer.loc13_34.1: type = class_type @Outer, @Outer(constants.%T) [symbolic = %Outer.loc13_34.2 (constants.%Outer.2)]
// CHECK:STDOUT:     %.loc13_37: @F.%Inner.type (%Inner.type.1) = specific_constant @Outer.%Inner.decl, @Outer(constants.%T) [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:     %Inner.ref: @F.%Inner.type (%Inner.type.1) = name_ref Inner, %.loc13_37 [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:     %U.ref.loc13_44: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %Inner.loc13_43.1: type = class_type @Inner, @Inner(constants.%T, constants.%U) [symbolic = %Inner.loc13_43.2 (constants.%Inner.2)]
// CHECK:STDOUT:     %p.param: @F.%Inner.loc13_43.2 (%Inner.2) = param p, runtime_param0
// CHECK:STDOUT:     %p: @F.%Inner.loc13_43.2 (%Inner.2) = bind_name p, %p.param
// CHECK:STDOUT:     %T.ref.loc13_52: type = name_ref T, %T.loc13_6.1 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:     %U.ref.loc13_55: type = name_ref U, %U.loc13_16.1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:     %.loc13_56.1: %.5 = tuple_literal (%T.ref.loc13_52, %U.ref.loc13_55)
// CHECK:STDOUT:     %.loc13_56.2: type = converted %.loc13_56.1, constants.%.6 [symbolic = %.loc13_56.3 (constants.%.6)]
// CHECK:STDOUT:     %return: ref @F.%.loc13_56.3 (%.6) = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {
// CHECK:STDOUT:     %p.patt: %Inner.4 = binding_pattern p
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %Outer.ref: %Outer.type = name_ref Outer, file.%Outer.decl [template = constants.%Outer.1]
// CHECK:STDOUT:     %C.ref.loc15_15: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %Outer: type = class_type @Outer, @Outer(constants.%C) [template = constants.%Outer.3]
// CHECK:STDOUT:     %.loc15_17: %Inner.type.2 = specific_constant @Outer.%Inner.decl, @Outer(constants.%C) [template = constants.%Inner.3]
// CHECK:STDOUT:     %Inner.ref: %Inner.type.2 = name_ref Inner, %.loc15_17 [template = constants.%Inner.3]
// CHECK:STDOUT:     %D.ref.loc15_24: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %Inner: type = class_type @Inner, @Inner(constants.%C, constants.%D) [template = constants.%Inner.4]
// CHECK:STDOUT:     %p.param: %Inner.4 = param p, runtime_param0
// CHECK:STDOUT:     %p: %Inner.4 = bind_name p, %p.param
// CHECK:STDOUT:     %C.ref.loc15_32: type = name_ref C, file.%C.decl [template = constants.%C]
// CHECK:STDOUT:     %D.ref.loc15_35: type = name_ref D, file.%D.decl [template = constants.%D]
// CHECK:STDOUT:     %.loc15_36.1: %.5 = tuple_literal (%C.ref.loc15_32, %D.ref.loc15_35)
// CHECK:STDOUT:     %.loc15_36.2: type = converted %.loc15_36.1, constants.%.7 [template = constants.%.7]
// CHECK:STDOUT:     %return: ref %.7 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Outer(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc4_13.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner: @Outer.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner (constants.%Inner.1)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Inner.decl: @Outer.%Inner.type (%Inner.type.1) = class_decl @Inner [symbolic = @Outer.%Inner (constants.%Inner.1)] {
// CHECK:STDOUT:       %U.patt: type = symbolic_binding_pattern U, 1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %U.param: type = param U, runtime_param<invalid>
// CHECK:STDOUT:       %U.loc5_15.1: type = bind_symbolic_name U, 1, %U.param [symbolic = %U.loc5_15.2 (constants.%U)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc6: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Outer.2
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
// CHECK:STDOUT:     %.loc5: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Inner.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %.loc8: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @D {
// CHECK:STDOUT:   %.loc9: <witness> = complete_type_witness %.2 [template = constants.%.3]
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%D
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc13_6.1: type, %U.loc13_16.1: type) {
// CHECK:STDOUT:   %T.loc13_6.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc13_6.2 (constants.%T)]
// CHECK:STDOUT:   %U.loc13_16.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc13_16.2 (constants.%U)]
// CHECK:STDOUT:   %Outer.loc13_34.2: type = class_type @Outer, @Outer(%T.loc13_6.2) [symbolic = %Outer.loc13_34.2 (constants.%Outer.2)]
// CHECK:STDOUT:   %Inner.type: type = generic_class_type @Inner, @Outer(%T.loc13_6.2) [symbolic = %Inner.type (constants.%Inner.type.1)]
// CHECK:STDOUT:   %Inner.loc13_37: @F.%Inner.type (%Inner.type.1) = struct_value () [symbolic = %Inner.loc13_37 (constants.%Inner.1)]
// CHECK:STDOUT:   %Inner.loc13_43.2: type = class_type @Inner, @Inner(%T.loc13_6.2, %U.loc13_16.2) [symbolic = %Inner.loc13_43.2 (constants.%Inner.2)]
// CHECK:STDOUT:   %.loc13_56.3: type = tuple_type (@F.%T.loc13_6.2 (%T), @F.%U.loc13_16.2 (%U)) [symbolic = %.loc13_56.3 (constants.%.6)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%T.loc13_6.1: type, %U.loc13_16.1: type](%p: @F.%Inner.loc13_43.2 (%Inner.2)) -> @F.%.loc13_56.3 (%.6);
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p: %Inner.4) -> %return: %.7 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %p.ref: %Inner.4 = name_ref p, %p
// CHECK:STDOUT:   %.loc15_28: ref %.7 = splice_block %return {}
// CHECK:STDOUT:   %F.call: init %.7 = call %F.ref(%p.ref) to %.loc15_28
// CHECK:STDOUT:   return %F.call to %return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner => constants.%Inner.1
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(constants.%T, constants.%U) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@Outer.%T.loc4_13.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(@F.%T.loc13_6.2) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Inner(@F.%T.loc13_6.2, @F.%U.loc13_16.2) {
// CHECK:STDOUT:   %U.loc5_15.2 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc13_6.2 => constants.%T
// CHECK:STDOUT:   %U.loc13_16.2 => constants.%U
// CHECK:STDOUT:   %Outer.loc13_34.2 => constants.%Outer.2
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.1
// CHECK:STDOUT:   %Inner.loc13_37 => constants.%Inner.1
// CHECK:STDOUT:   %Inner.loc13_43.2 => constants.%Inner.2
// CHECK:STDOUT:   %.loc13_56.3 => constants.%.6
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Outer(constants.%C) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%C
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner => constants.%Inner.3
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
// CHECK:STDOUT:   %Outer.loc13_34.2 => constants.%Outer.3
// CHECK:STDOUT:   %Inner.type => constants.%Inner.type.2
// CHECK:STDOUT:   %Inner.loc13_37 => constants.%Inner.3
// CHECK:STDOUT:   %Inner.loc13_43.2 => constants.%Inner.4
// CHECK:STDOUT:   %.loc13_56.3 => constants.%.7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nontype.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Int32.type: type = fn_type @Int32 [template]
// CHECK:STDOUT:   %.1: type = tuple_type () [template]
// CHECK:STDOUT:   %Int32: %Int32.type = struct_value () [template]
// CHECK:STDOUT:   %N: i32 = bind_symbolic_name N, 0 [symbolic]
// CHECK:STDOUT:   %WithNontype.type: type = generic_class_type @WithNontype [template]
// CHECK:STDOUT:   %WithNontype.1: %WithNontype.type = struct_value () [template]
// CHECK:STDOUT:   %WithNontype.2: type = class_type @WithNontype, @WithNontype(%N) [symbolic]
// CHECK:STDOUT:   %.2: type = struct_type {} [template]
// CHECK:STDOUT:   %.3: <witness> = complete_type_witness %.2 [template]
// CHECK:STDOUT:   %F.type: type = fn_type @F [template]
// CHECK:STDOUT:   %F: %F.type = struct_value () [template]
// CHECK:STDOUT:   %.4: type = ptr_type %.2 [template]
// CHECK:STDOUT:   %G.type: type = fn_type @G [template]
// CHECK:STDOUT:   %G: %G.type = struct_value () [template]
// CHECK:STDOUT:   %.5: i32 = int_literal 0 [template]
// CHECK:STDOUT:   %WithNontype.3: type = class_type @WithNontype, @WithNontype(%.5) [template]
// CHECK:STDOUT:   %struct: %WithNontype.3 = struct_value () [template]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [template] {
// CHECK:STDOUT:     .Int32 = %import_ref
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/operators
// CHECK:STDOUT:     import Core//prelude/types
// CHECK:STDOUT:     import Core//prelude/operators/arithmetic
// CHECK:STDOUT:     import Core//prelude/operators/as
// CHECK:STDOUT:     import Core//prelude/operators/bitwise
// CHECK:STDOUT:     import Core//prelude/operators/comparison
// CHECK:STDOUT:     import Core//prelude/types/bool
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %import_ref: %Int32.type = import_ref Core//prelude/types, inst+4, loaded [template = constants.%Int32]
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
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N, 0
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc4_23.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc4_23.2: type = converted %int.make_type_32, %.loc4_23.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N, runtime_param<invalid>
// CHECK:STDOUT:     %N.loc4_19.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc4_19.2 (constants.%N)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [template = constants.%F] {
// CHECK:STDOUT:     %N.patt: i32 = symbolic_binding_pattern N, 0
// CHECK:STDOUT:     %x.patt: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = binding_pattern x
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %int.make_type_32.loc6_10: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_10.1: type = value_of_initializer %int.make_type_32.loc6_10 [template = i32]
// CHECK:STDOUT:     %.loc6_10.2: type = converted %int.make_type_32.loc6_10, %.loc6_10.1 [template = i32]
// CHECK:STDOUT:     %N.param: i32 = param N, runtime_param<invalid>
// CHECK:STDOUT:     %N.loc6_6.1: i32 = bind_symbolic_name N, 0, %N.param [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [template = constants.%WithNontype.1]
// CHECK:STDOUT:     %N.ref.loc6_30: i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     %WithNontype.loc6_29.1: type = class_type @WithNontype, @WithNontype(constants.%N) [symbolic = %WithNontype.loc6_29.2 (constants.%WithNontype.2)]
// CHECK:STDOUT:     %x.param: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = param x, runtime_param0
// CHECK:STDOUT:     %x: @F.%WithNontype.loc6_29.2 (%WithNontype.2) = bind_name x, %x.param
// CHECK:STDOUT:     %int.make_type_32.loc6_37: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc6_37.1: type = value_of_initializer %int.make_type_32.loc6_37 [template = i32]
// CHECK:STDOUT:     %.loc6_37.2: type = converted %int.make_type_32.loc6_37, %.loc6_37.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [template = constants.%G] {} {
// CHECK:STDOUT:     %int.make_type_32: init type = call constants.%Int32() [template = i32]
// CHECK:STDOUT:     %.loc8_11.1: type = value_of_initializer %int.make_type_32 [template = i32]
// CHECK:STDOUT:     %.loc8_11.2: type = converted %int.make_type_32, %.loc8_11.1 [template = i32]
// CHECK:STDOUT:     %return: ref i32 = var <return slot>
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @WithNontype(%N.loc4_19.1: i32) {
// CHECK:STDOUT:   %N.loc4_19.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc4_19.2 (constants.%N)]
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
// CHECK:STDOUT: fn @Int32() -> type = "int.make_type_32";
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%N.loc6_6.1: i32) {
// CHECK:STDOUT:   %N.loc6_6.2: i32 = bind_symbolic_name N, 0 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:   %WithNontype.loc6_29.2: type = class_type @WithNontype, @WithNontype(%N.loc6_6.2) [symbolic = %WithNontype.loc6_29.2 (constants.%WithNontype.2)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%N.loc6_6.1: i32](%x: @F.%WithNontype.loc6_29.2 (%WithNontype.2)) -> i32 {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %N.ref.loc6_50: i32 = name_ref N, %N.loc6_6.1 [symbolic = %N.loc6_6.2 (constants.%N)]
// CHECK:STDOUT:     return %N.ref.loc6_50
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G() -> i32 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %F.ref: %F.type = name_ref F, file.%F.decl [template = constants.%F]
// CHECK:STDOUT:   %.loc9_13.1: %.2 = struct_literal ()
// CHECK:STDOUT:   %WithNontype.ref: %WithNontype.type = name_ref WithNontype, file.%WithNontype.decl [template = constants.%WithNontype.1]
// CHECK:STDOUT:   %.loc9_30: i32 = int_literal 0 [template = constants.%.5]
// CHECK:STDOUT:   %WithNontype: type = class_type @WithNontype, @WithNontype(constants.%.5) [template = constants.%WithNontype.3]
// CHECK:STDOUT:   %.loc9_13.2: ref %WithNontype.3 = temporary_storage
// CHECK:STDOUT:   %.loc9_13.3: init %WithNontype.3 = class_init (), %.loc9_13.2 [template = constants.%struct]
// CHECK:STDOUT:   %.loc9_13.4: ref %WithNontype.3 = temporary %.loc9_13.2, %.loc9_13.3
// CHECK:STDOUT:   %.loc9_15.1: ref %WithNontype.3 = converted %.loc9_13.1, %.loc9_13.4
// CHECK:STDOUT:   %.loc9_15.2: %WithNontype.3 = bind_value %.loc9_15.1
// CHECK:STDOUT:   %F.call: init i32 = call %F.ref(%.loc9_15.2)
// CHECK:STDOUT:   %.loc9_33.1: i32 = value_of_initializer %F.call
// CHECK:STDOUT:   %.loc9_33.2: i32 = converted %F.call, %.loc9_33.1
// CHECK:STDOUT:   return %.loc9_33.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%N) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(@F.%N.loc6_6.2) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%N
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%N) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%N
// CHECK:STDOUT:   %WithNontype.loc6_29.2 => constants.%WithNontype.2
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WithNontype(constants.%.5) {
// CHECK:STDOUT:   %N.loc4_19.2 => constants.%.5
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%.5) {
// CHECK:STDOUT:   %N.loc6_6.2 => constants.%.5
// CHECK:STDOUT:   %WithNontype.loc6_29.2 => constants.%WithNontype.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
