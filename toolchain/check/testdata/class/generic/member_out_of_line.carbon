// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/generic/member_out_of_line.carbon

// --- basic.carbon

library "[[@TEST_NAME]]";

class Class(T:! type) {
  fn F(n: T) -> T;
  fn G[self: Self]() -> T;
  var n: T;
}

fn Class(T:! type).F(n: T) -> T {
  return n;
}

fn Class(T:! type).G[self: Self]() -> T {
  return self.n;
}

// --- nested.carbon

library "[[@TEST_NAME]]";

class A(T:! type) {
  class B(N:! T) {
    fn F[self: Self](a: T);
  }
}

fn A(T:! type).B(N:! T).F[self: Self](a: T) {}

// --- fail_mismatched_not_generic_vs_generic.carbon

library "[[@TEST_NAME]]";

class NotGeneric {
  fn F();
}

// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter list [RedeclParamListDiffers]
// CHECK:STDERR: fn NotGeneric(T:! type).F() {}
// CHECK:STDERR:    ^~~~~~~~~~
// CHECK:STDERR: fail_mismatched_not_generic_vs_generic.carbon:[[@LINE-7]]:1: note: previously declared without parameter list [RedeclParamListPrevious]
// CHECK:STDERR: class NotGeneric {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn NotGeneric(T:! type).F() {}

// --- fail_mismatched_too_few_args.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn TooFew();
}

// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic().TooFew() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_few_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic().TooFew() {}

// --- fail_mismatched_too_many_args.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn TooMany();
}

// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE+7]]:4: error: redeclaration differs because of parameter count of 2 [RedeclParamCountDiffers]
// CHECK:STDERR: fn Generic(T:! type, U:! type).TooMany() {}
// CHECK:STDERR:    ^~~~~~~
// CHECK:STDERR: fail_mismatched_too_many_args.carbon:[[@LINE-7]]:1: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR: ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn Generic(T:! type, U:! type).TooMany() {}

// --- fail_mismatched_wrong_arg_type.carbon

library "[[@TEST_NAME]]";

class Generic(T:! type) {
  fn WrongType();
}

// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE+7]]:12: error: type `()` of parameter 1 in redeclaration differs from previous parameter type `type` [RedeclParamDiffersType]
// CHECK:STDERR: fn Generic(T:! ()).WrongType() {}
// CHECK:STDERR:            ^
// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE-7]]:15: note: previous declaration's corresponding parameter here [RedeclParamPrevious]
// CHECK:STDERR: class Generic(T:! type) {
// CHECK:STDERR:               ^
// CHECK:STDERR:
fn Generic(T:! ()).WrongType() {}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T) [symbolic]
// CHECK:STDOUT:   %G: %G.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: %T} [symbolic]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %struct_type.n [symbolic]
// CHECK:STDOUT:   %require_complete.4f8: <witness> = require_complete_type %Class [symbolic]
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
// CHECK:STDOUT:     .Class = %Class.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Class.decl: %Class.type = class_decl @Class [concrete = constants.%Class.generic] {
// CHECK:STDOUT:     %T.patt.loc4_13.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %n.patt: @F.%T.loc5 (%T) = binding_pattern n
// CHECK:STDOUT:     %n.param_patt: @F.%T.loc5 (%T) = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:     %return.patt: @F.%T.loc5 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @F.%T.loc5 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc10_31: type = name_ref T, %T.loc10 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:     %n.param.loc10: @F.%T.loc5 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc10_25: type = name_ref T, %T.loc10 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:     %n.loc10: @F.%T.loc5 (%T) = bind_name n, %n.param.loc10
// CHECK:STDOUT:     %return.param.loc10: ref @F.%T.loc5 (%T) = out_param call_param1
// CHECK:STDOUT:     %return.loc10: ref @F.%T.loc5 (%T) = return_slot %return.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl: @Class.%G.type (%G.type) = fn_decl @G [symbolic = constants.%G] {
// CHECK:STDOUT:     %self.patt: @G.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @G.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %return.patt: @G.%T.loc6 (%T) = return_slot_pattern
// CHECK:STDOUT:     %return.param_patt: @G.%T.loc6 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14: type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc14 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:     %self.param.loc14: @G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %Self.ref.loc14 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc14_28.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc14: type = name_ref Self, %.loc14_28.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc14: @G.%Class (%Class) = bind_name self, %self.param.loc14
// CHECK:STDOUT:     %return.param.loc14: ref @G.%T.loc6 (%T) = out_param call_param1
// CHECK:STDOUT:     %return.loc14: ref @G.%T.loc6 (%T) = return_slot %return.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_13.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_13.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @Class(%T.loc4_13.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @Class.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:   %G.type: type = fn_type @G, @Class(%T.loc4_13.2) [symbolic = %G.type (constants.%G.type)]
// CHECK:STDOUT:   %G: @Class.%G.type (%G.type) = struct_value () [symbolic = %G (constants.%G)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @Class.%T.loc4_13.2 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.2) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @Class.%Class (%Class), @Class.%T.loc4_13.2 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n: type = struct_type {.n: @Class.%T.loc4_13.2 (%T)} [symbolic = %struct_type.n (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc8_1.2: <witness> = complete_type_witness @Class.%struct_type.n (%struct_type.n) [symbolic = %complete_type.loc8_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @Class.%F.type (%F.type) = fn_decl @F [symbolic = @Class.%F (constants.%F)] {
// CHECK:STDOUT:       %n.patt: @F.%T.loc5 (%T) = binding_pattern n
// CHECK:STDOUT:       %n.param_patt: @F.%T.loc5 (%T) = value_param_pattern %n.patt, call_param0
// CHECK:STDOUT:       %return.patt: @F.%T.loc5 (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @F.%T.loc5 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc5_17: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:       %n.param.loc5: @F.%T.loc5 (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc5_11: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:       %n.loc5: @F.%T.loc5 (%T) = bind_name n, %n.param.loc5
// CHECK:STDOUT:       %return.param.loc5: ref @F.%T.loc5 (%T) = out_param call_param1
// CHECK:STDOUT:       %return.loc5: ref @F.%T.loc5 (%T) = return_slot %return.param.loc5
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %G.decl: @Class.%G.type (%G.type) = fn_decl @G [symbolic = @Class.%G (constants.%G)] {
// CHECK:STDOUT:       %self.patt: @G.%Class (%Class) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @G.%Class (%Class) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %return.patt: @G.%T.loc6 (%T) = return_slot_pattern
// CHECK:STDOUT:       %return.param_patt: @G.%T.loc6 (%T) = out_param_pattern %return.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, @Class.%T.loc4_13.1 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:       %self.param.loc6: @G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc6_14.1: type = splice_block %Self.ref.loc6 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc6_14.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc6: type = name_ref Self, %.loc6_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc6: @G.%Class (%Class) = bind_name self, %self.param.loc6
// CHECK:STDOUT:       %return.param.loc6: ref @G.%T.loc6 (%T) = out_param call_param1
// CHECK:STDOUT:       %return.loc6: ref @G.%T.loc6 (%T) = return_slot %return.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc7_8: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     name_binding_decl {
// CHECK:STDOUT:       %.loc7_3: @Class.%Class.elem (%Class.elem) = var_pattern %.loc7_8
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.var: ref @Class.%Class.elem (%Class.elem) = var <none>
// CHECK:STDOUT:     %complete_type.loc8_1.1: <witness> = complete_type_witness %struct_type.n [symbolic = %complete_type.loc8_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:     .G = %G.decl
// CHECK:STDOUT:     .n = %.loc7_8
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc5: type = bind_symbolic_name T, 0 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type @F.%T.loc5 (%T) [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param_patt: @F.%T.loc5 (%T)) -> @F.%T.loc5 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @F.%T.loc5 (%T) = name_ref n, %n.loc10
// CHECK:STDOUT:     return %n.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc6) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type @G.%Class (%Class) [symbolic = %require_complete.loc14 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type @G.%Class (%Class), @G.%T.loc6 (%T) [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type @G.%T.loc6 (%T) [symbolic = %require_complete.loc15 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @G.%Class (%Class)]() -> @G.%T.loc6 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @G.%Class (%Class) = name_ref self, %self.loc14
// CHECK:STDOUT:     %n.ref: @G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc7_8 [concrete = @Class.%.loc7_8]
// CHECK:STDOUT:     %.loc15_14.1: ref @G.%T.loc6 (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc15_14.2: @G.%T.loc6 (%T) = bind_value %.loc15_14.1
// CHECK:STDOUT:     return %.loc15_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_13.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT:   %G.type => constants.%G.type
// CHECK:STDOUT:   %G => constants.%G
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc5 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @G(constants.%T) {
// CHECK:STDOUT:   %T.loc6 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(@G.%T.loc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(%T.loc4_13.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %N.patt: %T = symbolic_binding_pattern N, 1 [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T) [symbolic]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %require_complete.fca: <witness> = require_complete_type %B [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
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
// CHECK:STDOUT:     .A = %A.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %A.decl: %A.type = class_decl @A [concrete = constants.%A.generic] {
// CHECK:STDOUT:     %T.patt.loc4_9.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: @B.%F.type (%F.type) = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%B (%B) = binding_pattern self
// CHECK:STDOUT:     %self.param_patt: @F.%B (%B) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:     %a.patt: @F.%T.loc6 (%T) = binding_pattern a
// CHECK:STDOUT:     %a.param_patt: @F.%T.loc6 (%T) = value_param_pattern %a.patt, call_param1
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0 [symbolic = @A.%T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc10 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     %N.loc10: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = @B.%N.loc5_11.2 (constants.%N)]
// CHECK:STDOUT:     %self.param.loc10: @F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc10_33.1: type = splice_block %Self.ref.loc10 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc10_33.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc10: type = name_ref Self, %.loc10_33.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc10: @F.%B (%B) = bind_name self, %self.param.loc10
// CHECK:STDOUT:     %a.param.loc10: @F.%T.loc6 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc10_42: type = name_ref T, %T.loc10 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:     %a.loc10: @F.%T.loc6 (%T) = bind_name a, %a.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc4_9.1: type) {
// CHECK:STDOUT:   %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_9.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_9.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T.loc4_9.2) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B.generic: @A.%B.type (%B.type) = struct_value () [symbolic = %B.generic (constants.%B.generic)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: @A.%B.type (%B.type) = class_decl @B [symbolic = @A.%B.generic (constants.%B.generic)] {
// CHECK:STDOUT:       %N.patt.loc5_11.1: @B.%T (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc5_11.2 (constants.%N.patt)]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc4_9.1 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %N.loc5_11.1: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc5_11.2 (constants.%N)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%A
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .B = %B.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @B(@A.%T.loc4_9.1: type, %N.loc5_11.1: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.loc5_11.2: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc5_11.2 (constants.%N)]
// CHECK:STDOUT:   %N.patt.loc5_11.2: @B.%T (%T) = symbolic_binding_pattern N, 1 [symbolic = %N.patt.loc5_11.2 (constants.%N.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type: type = fn_type @F, @B(%T, %N.loc5_11.2) [symbolic = %F.type (constants.%F.type)]
// CHECK:STDOUT:   %F: @B.%F.type (%F.type) = struct_value () [symbolic = %F (constants.%F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %F.decl: @B.%F.type (%F.type) = fn_decl @F [symbolic = @B.%F (constants.%F)] {
// CHECK:STDOUT:       %self.patt: @F.%B (%B) = binding_pattern self
// CHECK:STDOUT:       %self.param_patt: @F.%B (%B) = value_param_pattern %self.patt, call_param0
// CHECK:STDOUT:       %a.patt: @F.%T.loc6 (%T) = binding_pattern a
// CHECK:STDOUT:       %a.param_patt: @F.%T.loc6 (%T) = value_param_pattern %a.patt, call_param1
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc6: @F.%B (%B) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %Self.ref.loc6 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:         %.loc6_16.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:         %Self.ref.loc6: type = name_ref Self, %.loc6_16.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc6: @F.%B (%B) = bind_name self, %self.param.loc6
// CHECK:STDOUT:       %a.param.loc6: @F.%T.loc6 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, @A.%T.loc4_9.1 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:       %a.loc6: @F.%T.loc6 (%T) = bind_name a, %a.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@A.%T.loc4_9.1: type, @B.%N.loc5_11.1: @B.%T (%T)) {
// CHECK:STDOUT:   %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:   %N.loc6: @F.%T.loc6 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc6 (constants.%N)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc6, %N.loc6) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_31: <witness> = require_complete_type @F.%B (%B) [symbolic = %require_complete.loc10_31 (constants.%require_complete.fca)]
// CHECK:STDOUT:   %require_complete.loc10_40: <witness> = require_complete_type @F.%T.loc6 (%T) [symbolic = %require_complete.loc10_40 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn[%self.param_patt: @F.%B (%B)](%a.param_patt: @F.%T.loc6 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_9.2 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B.generic => constants.%B.generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.loc5_11.2 => constants.%N
// CHECK:STDOUT:   %N.patt.loc5_11.2 => constants.%N
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %F.type => constants.%F.type
// CHECK:STDOUT:   %F => constants.%F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6 => constants.%T
// CHECK:STDOUT:   %N.loc6 => constants.%N
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(@F.%T.loc6, @F.%N.loc6) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(%T, %N.loc5_11.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(%T.loc4_9.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_not_generic_vs_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric: type = class_type @NotGeneric [concrete]
// CHECK:STDOUT:   %F.type.065: type = fn_type @F.1 [concrete]
// CHECK:STDOUT:   %F.fdb: %F.type.065 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %F.type.b25: type = fn_type @F.2 [concrete]
// CHECK:STDOUT:   %F.c41: %F.type.b25 = struct_value () [concrete]
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
// CHECK:STDOUT:     .NotGeneric = %NotGeneric.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %NotGeneric.decl: type = class_decl @NotGeneric [concrete = constants.%NotGeneric] {} {}
// CHECK:STDOUT:   %F.decl: %F.type.b25 = fn_decl @F.2 [concrete = constants.%F.c41] {} {
// CHECK:STDOUT:     %T.loc15_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGeneric {
// CHECK:STDOUT:   %F.decl: %F.type.065 = fn_decl @F.1 [concrete = constants.%F.fdb] {} {}
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGeneric
// CHECK:STDOUT:   .F = %F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.1();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F.2(%T.loc15_15.1: type) {
// CHECK:STDOUT:   %T.loc15_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F.2(constants.%T) {
// CHECK:STDOUT:   %T.loc15_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooFew.type.dac: type = fn_type @TooFew.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooFew.840: %TooFew.type.dac = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TooFew.type.e89: type = fn_type @TooFew.2 [concrete]
// CHECK:STDOUT:   %TooFew.7c3: %TooFew.type.e89 = struct_value () [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type.e89 = fn_decl @TooFew.2 [concrete = constants.%TooFew.7c3] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew.1, @Generic(%T.loc4_15.2) [symbolic = %TooFew.type (constants.%TooFew.type.dac)]
// CHECK:STDOUT:   %TooFew: @Generic.%TooFew.type (%TooFew.type.dac) = struct_value () [symbolic = %TooFew (constants.%TooFew.840)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %TooFew.decl: @Generic.%TooFew.type (%TooFew.type.dac) = fn_decl @TooFew.1 [symbolic = @Generic.%TooFew (constants.%TooFew.840)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .TooFew = %TooFew.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooFew.1(@Generic.%T.loc4_15.1: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew.2() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooFew.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_many_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooMany.type.cc8: type = fn_type @TooMany.1, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %TooMany.dc4: %TooMany.type.cc8 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic]
// CHECK:STDOUT:   %TooMany.type.c84: type = fn_type @TooMany.2 [concrete]
// CHECK:STDOUT:   %TooMany.1dc: %TooMany.type.c84 = struct_value () [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type.c84 = fn_decl @TooMany.2 [concrete = constants.%TooMany.1dc] {} {
// CHECK:STDOUT:     %T.loc15_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T)]
// CHECK:STDOUT:     %U.loc15_22.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_22.2 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany.1, @Generic(%T.loc4_15.2) [symbolic = %TooMany.type (constants.%TooMany.type.cc8)]
// CHECK:STDOUT:   %TooMany: @Generic.%TooMany.type (%TooMany.type.cc8) = struct_value () [symbolic = %TooMany (constants.%TooMany.dc4)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %TooMany.decl: @Generic.%TooMany.type (%TooMany.type.cc8) = fn_decl @TooMany.1 [symbolic = @Generic.%TooMany (constants.%TooMany.dc4)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .TooMany = %TooMany.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooMany.1(@Generic.%T.loc4_15.1: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooMany.2(%T.loc15_12.1: type, %U.loc15_22.1: type) {
// CHECK:STDOUT:   %T.loc15_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T)]
// CHECK:STDOUT:   %T.patt: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt)]
// CHECK:STDOUT:   %U.loc15_22.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_22.2 (constants.%U)]
// CHECK:STDOUT:   %U.patt: type = symbolic_binding_pattern U, 1 [symbolic = %U.patt (constants.%U.patt)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooMany.1(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooMany.2(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%T
// CHECK:STDOUT:   %T.patt => constants.%T
// CHECK:STDOUT:   %U.loc15_22.2 => constants.%U
// CHECK:STDOUT:   %U.patt => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_wrong_arg_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e01: type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T.8b3) [symbolic]
// CHECK:STDOUT:   %WrongType.type.c41: type = fn_type @WrongType.1, @Generic(%T.8b3) [symbolic]
// CHECK:STDOUT:   %WrongType.408: %WrongType.type.c41 = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.7a6: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %T.patt.e60: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic]
// CHECK:STDOUT:   %WrongType.type.edf: type = fn_type @WrongType.2 [concrete]
// CHECK:STDOUT:   %WrongType.131: %WrongType.type.edf = struct_value () [concrete]
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
// CHECK:STDOUT:     .Generic = %Generic.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Generic.decl: %Generic.type = class_decl @Generic [concrete = constants.%Generic.generic] {
// CHECK:STDOUT:     %T.patt.loc4_15.1: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongType.decl: %WrongType.type.edf = fn_decl @WrongType.2 [concrete = constants.%WrongType.131] {} {
// CHECK:STDOUT:     %.loc15_17.1: type = splice_block %.loc15_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_17.3: type = converted %.loc15_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_12.1: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T.7a6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.1: type) {
// CHECK:STDOUT:   %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.2 (constants.%T.8b3)]
// CHECK:STDOUT:   %T.patt.loc4_15.2: type = symbolic_binding_pattern T, 0 [symbolic = %T.patt.loc4_15.2 (constants.%T.patt.e01)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %WrongType.type: type = fn_type @WrongType.1, @Generic(%T.loc4_15.2) [symbolic = %WrongType.type (constants.%WrongType.type.c41)]
// CHECK:STDOUT:   %WrongType: @Generic.%WrongType.type (%WrongType.type.c41) = struct_value () [symbolic = %WrongType (constants.%WrongType.408)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %WrongType.decl: @Generic.%WrongType.type (%WrongType.type.c41) = fn_decl @WrongType.1 [symbolic = @Generic.%WrongType (constants.%WrongType.408)] {} {}
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .WrongType = %WrongType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongType.1(@Generic.%T.loc4_15.1: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongType.2(%T.loc15_12.1: %empty_tuple.type) {
// CHECK:STDOUT:   %T.loc15_12.2: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.2 (constants.%T.7a6)]
// CHECK:STDOUT:   %T.patt: %empty_tuple.type = symbolic_binding_pattern T, 0 [symbolic = %T.patt (constants.%T.patt.e60)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T.8b3) {
// CHECK:STDOUT:   %T.loc4_15.2 => constants.%T.8b3
// CHECK:STDOUT:   %T.patt.loc4_15.2 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongType.1(constants.%T.8b3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(%T.loc4_15.2) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongType.2(constants.%T.7a6) {
// CHECK:STDOUT:   %T.loc15_12.2 => constants.%T.7a6
// CHECK:STDOUT:   %T.patt => constants.%T.7a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
