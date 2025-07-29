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

// CHECK:STDERR: fail_mismatched_wrong_arg_type.carbon:[[@LINE+7]]:12: error: type `<pattern for ()>` of parameter 1 in redeclaration differs from previous parameter type `<pattern for type>` [RedeclParamDiffersType]
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
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Class.type: type = generic_class_type @Class [concrete]
// CHECK:STDOUT:   %Class.generic: %Class.type = struct_value () [concrete]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.F: %Class.F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %pattern_type.3c1: type = pattern_type %Class [symbolic]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T) [symbolic]
// CHECK:STDOUT:   %Class.G: %Class.G.type = struct_value () [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_13.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.F.decl: %Class.F.type = fn_decl @Class.F [symbolic = constants.%Class.F] {
// CHECK:STDOUT:     %n.patt: @Class.F.%pattern_type (%pattern_type.7dc) = binding_pattern n [concrete]
// CHECK:STDOUT:     %n.param_patt: @Class.F.%pattern_type (%pattern_type.7dc) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Class.F.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.F.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc10_31: type = name_ref T, %T.loc10 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:     %n.param.loc10: @Class.F.%T.loc5 (%T) = value_param call_param0
// CHECK:STDOUT:     %T.ref.loc10_25: type = name_ref T, %T.loc10 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:     %n.loc10: @Class.F.%T.loc5 (%T) = bind_name n, %n.param.loc10
// CHECK:STDOUT:     %return.param.loc10: ref @Class.F.%T.loc5 (%T) = out_param call_param1
// CHECK:STDOUT:     %return.loc10: ref @Class.F.%T.loc5 (%T) = return_slot %return.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Class.G.decl: %Class.G.type = fn_decl @Class.G [symbolic = constants.%Class.G] {
// CHECK:STDOUT:     %self.patt: @Class.G.%pattern_type.loc6_8 (%pattern_type.3c1) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @Class.G.%pattern_type.loc6_8 (%pattern_type.3c1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: @Class.G.%pattern_type.loc6_22 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: @Class.G.%pattern_type.loc6_22 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc14: type = bind_symbolic_name T, 0 [symbolic = @Class.%T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc14: type = name_ref T, %T.loc14 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:     %self.param.loc14: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:     %.loc14_28.1: type = splice_block %Self.ref.loc14 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:       %.loc14_28.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       %Self.ref.loc14: type = name_ref Self, %.loc14_28.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc14: @Class.G.%Class (%Class) = bind_name self, %self.param.loc14
// CHECK:STDOUT:     %return.param.loc14: ref @Class.G.%T.loc6 (%T) = out_param call_param1
// CHECK:STDOUT:     %return.loc14: ref @Class.G.%T.loc6 (%T) = return_slot %return.param.loc14
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Class(%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc4_13.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type: type = fn_type @Class.F, @Class(%T.loc4_13.1) [symbolic = %Class.F.type (constants.%Class.F.type)]
// CHECK:STDOUT:   %Class.F: @Class.%Class.F.type (%Class.F.type) = struct_value () [symbolic = %Class.F (constants.%Class.F)]
// CHECK:STDOUT:   %Class.G.type: type = fn_type @Class.G, @Class(%T.loc4_13.1) [symbolic = %Class.G.type (constants.%Class.G.type)]
// CHECK:STDOUT:   %Class.G: @Class.%Class.G.type (%Class.G.type) = struct_value () [symbolic = %Class.G (constants.%Class.G)]
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc4_13.1 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc4_13.1) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc4_13.1 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %struct_type.n.loc8_1.2: type = struct_type {.n: @Class.%T.loc4_13.1 (%T)} [symbolic = %struct_type.n.loc8_1.2 (constants.%struct_type.n)]
// CHECK:STDOUT:   %complete_type.loc8_1.2: <witness> = complete_type_witness %struct_type.n.loc8_1.2 [symbolic = %complete_type.loc8_1.2 (constants.%complete_type)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Class.F.decl: @Class.%Class.F.type (%Class.F.type) = fn_decl @Class.F [symbolic = @Class.%Class.F (constants.%Class.F)] {
// CHECK:STDOUT:       %n.patt: @Class.F.%pattern_type (%pattern_type.7dc) = binding_pattern n [concrete]
// CHECK:STDOUT:       %n.param_patt: @Class.F.%pattern_type (%pattern_type.7dc) = value_param_pattern %n.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.F.%pattern_type (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.F.%pattern_type (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc5_17: type = name_ref T, @Class.%T.loc4_13.2 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:       %n.param.loc5: @Class.F.%T.loc5 (%T) = value_param call_param0
// CHECK:STDOUT:       %T.ref.loc5_11: type = name_ref T, @Class.%T.loc4_13.2 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:       %n.loc5: @Class.F.%T.loc5 (%T) = bind_name n, %n.param.loc5
// CHECK:STDOUT:       %return.param.loc5: ref @Class.F.%T.loc5 (%T) = out_param call_param1
// CHECK:STDOUT:       %return.loc5: ref @Class.F.%T.loc5 (%T) = return_slot %return.param.loc5
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %Class.G.decl: @Class.%Class.G.type (%Class.G.type) = fn_decl @Class.G [symbolic = @Class.%Class.G (constants.%Class.G)] {
// CHECK:STDOUT:       %self.patt: @Class.G.%pattern_type.loc6_8 (%pattern_type.3c1) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @Class.G.%pattern_type.loc6_8 (%pattern_type.3c1) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %return.patt: @Class.G.%pattern_type.loc6_22 (%pattern_type.7dc) = return_slot_pattern [concrete]
// CHECK:STDOUT:       %return.param_patt: @Class.G.%pattern_type.loc6_22 (%pattern_type.7dc) = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, @Class.%T.loc4_13.2 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:       %self.param.loc6: @Class.G.%Class (%Class) = value_param call_param0
// CHECK:STDOUT:       %.loc6_14.1: type = splice_block %Self.ref.loc6 [symbolic = %Class (constants.%Class)] {
// CHECK:STDOUT:         %.loc6_14.2: type = specific_constant constants.%Class, @Class(constants.%T) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:         %Self.ref.loc6: type = name_ref Self, %.loc6_14.2 [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc6: @Class.G.%Class (%Class) = bind_name self, %self.param.loc6
// CHECK:STDOUT:       %return.param.loc6: ref @Class.G.%T.loc6 (%T) = out_param call_param1
// CHECK:STDOUT:       %return.loc6: ref @Class.G.%T.loc6 (%T) = return_slot %return.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.ref: type = name_ref T, %T.loc4_13.2 [symbolic = %T.loc4_13.1 (constants.%T)]
// CHECK:STDOUT:     %.loc7: @Class.%Class.elem (%Class.elem) = field_decl n, element0 [concrete]
// CHECK:STDOUT:     %struct_type.n.loc8_1.1: type = struct_type {.n: %T} [symbolic = %struct_type.n.loc8_1.2 (constants.%struct_type.n)]
// CHECK:STDOUT:     %complete_type.loc8_1.1: <witness> = complete_type_witness %struct_type.n.loc8_1.1 [symbolic = %complete_type.loc8_1.2 (constants.%complete_type)]
// CHECK:STDOUT:     complete_type_witness = %complete_type.loc8_1.1
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Class
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %Class.F.decl
// CHECK:STDOUT:     .G = %Class.G.decl
// CHECK:STDOUT:     .n = %.loc7
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.F(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc5: type = bind_symbolic_name T, 0 [symbolic = %T.loc5 (constants.%T)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T.loc5 [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete: <witness> = require_complete_type %T.loc5 [symbolic = %require_complete (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param.loc10: @Class.F.%T.loc5 (%T)) -> @Class.F.%T.loc5 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @Class.F.%T.loc5 (%T) = name_ref n, %n.loc10
// CHECK:STDOUT:     return %n.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Class.G(@Class.%T.loc4_13.2: type) {
// CHECK:STDOUT:   %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T.loc6) [symbolic = %Class (constants.%Class)]
// CHECK:STDOUT:   %pattern_type.loc6_8: type = pattern_type %Class [symbolic = %pattern_type.loc6_8 (constants.%pattern_type.3c1)]
// CHECK:STDOUT:   %pattern_type.loc6_22: type = pattern_type %T.loc6 [symbolic = %pattern_type.loc6_22 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc14: <witness> = require_complete_type %Class [symbolic = %require_complete.loc14 (constants.%require_complete.4f8)]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc6 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc15: <witness> = require_complete_type %T.loc6 [symbolic = %require_complete.loc15 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc14: @Class.G.%Class (%Class)) -> @Class.G.%T.loc6 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @Class.G.%Class (%Class) = name_ref self, %self.loc14
// CHECK:STDOUT:     %n.ref: @Class.G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc7 [concrete = @Class.%.loc7]
// CHECK:STDOUT:     %.loc15_14.1: ref @Class.G.%T.loc6 (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc15_14.2: @Class.G.%T.loc6 (%T) = bind_value %.loc15_14.1
// CHECK:STDOUT:     return %.loc15_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class(constants.%T) {
// CHECK:STDOUT:   %T.loc4_13.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Class.F.type => constants.%Class.F.type
// CHECK:STDOUT:   %Class.F => constants.%Class.F
// CHECK:STDOUT:   %Class.G.type => constants.%Class.G.type
// CHECK:STDOUT:   %Class.G => constants.%Class.G
// CHECK:STDOUT:   %require_complete => constants.%require_complete.4ae
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %Class.elem => constants.%Class.elem
// CHECK:STDOUT:   %struct_type.n.loc8_1.2 => constants.%struct_type.n
// CHECK:STDOUT:   %complete_type.loc8_1.2 => constants.%complete_type
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.F(constants.%T) {
// CHECK:STDOUT:   %T.loc5 => constants.%T
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Class.G(constants.%T) {
// CHECK:STDOUT:   %T.loc6 => constants.%T
// CHECK:STDOUT:   %Class => constants.%Class
// CHECK:STDOUT:   %pattern_type.loc6_8 => constants.%pattern_type.3c1
// CHECK:STDOUT:   %pattern_type.loc6_22 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %A.type: type = generic_class_type @A [concrete]
// CHECK:STDOUT:   %A.generic: %A.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A, @A(%T) [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T) [symbolic]
// CHECK:STDOUT:   %B.generic: %B.type = struct_value () [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %pattern_type.13e: type = pattern_type %B [symbolic]
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %B.F: %B.F.type = struct_value () [symbolic]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_9.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %B.F.decl: %B.F.type = fn_decl @B.F [symbolic = constants.%B.F] {
// CHECK:STDOUT:     %self.patt: @B.F.%pattern_type.loc6_10 (%pattern_type.13e) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @B.F.%pattern_type.loc6_10 (%pattern_type.13e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @B.F.%pattern_type.loc6_22 (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @B.F.%pattern_type.loc6_22 (%pattern_type.7dc) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc10: type = bind_symbolic_name T, 0 [symbolic = @A.%T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc10_22: type = name_ref T, %T.loc10 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     %N.loc10: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = @B.%N.loc5_11.1 (constants.%N)]
// CHECK:STDOUT:     %self.param.loc10: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc10_33.1: type = splice_block %Self.ref.loc10 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc10_33.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc10: type = name_ref Self, %.loc10_33.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc10: @B.F.%B (%B) = bind_name self, %self.param.loc10
// CHECK:STDOUT:     %a.param.loc10: @B.F.%T.loc6 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc10_42: type = name_ref T, %T.loc10 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:     %a.loc10: @B.F.%T.loc6 (%T) = bind_name a, %a.param.loc10
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @A(%T.loc4_9.2: type) {
// CHECK:STDOUT:   %T.loc4_9.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_9.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type: type = generic_class_type @B, @A(%T.loc4_9.1) [symbolic = %B.type (constants.%B.type)]
// CHECK:STDOUT:   %B.generic: @A.%B.type (%B.type) = struct_value () [symbolic = %B.generic (constants.%B.generic)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.decl: @A.%B.type (%B.type) = class_decl @B [symbolic = @A.%B.generic (constants.%B.generic)] {
// CHECK:STDOUT:       %N.patt: @B.%pattern_type (%pattern_type.7dc) = symbolic_binding_pattern N, 1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %T.ref: type = name_ref T, @A.%T.loc4_9.2 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:       %N.loc5_11.2: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc5_11.1 (constants.%N)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
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
// CHECK:STDOUT: generic class @B(@A.%T.loc4_9.2: type, %N.loc5_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:   %N.loc5_11.1: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc5_11.1 (constants.%N)]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %T [symbolic = %pattern_type (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type: type = fn_type @B.F, @B(%T, %N.loc5_11.1) [symbolic = %B.F.type (constants.%B.F.type)]
// CHECK:STDOUT:   %B.F: @B.%B.F.type (%B.F.type) = struct_value () [symbolic = %B.F (constants.%B.F)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %B.F.decl: @B.%B.F.type (%B.F.type) = fn_decl @B.F [symbolic = @B.%B.F (constants.%B.F)] {
// CHECK:STDOUT:       %self.patt: @B.F.%pattern_type.loc6_10 (%pattern_type.13e) = binding_pattern self [concrete]
// CHECK:STDOUT:       %self.param_patt: @B.F.%pattern_type.loc6_10 (%pattern_type.13e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:       %a.patt: @B.F.%pattern_type.loc6_22 (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:       %a.param_patt: @B.F.%pattern_type.loc6_22 (%pattern_type.7dc) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:     } {
// CHECK:STDOUT:       %self.param.loc6: @B.F.%B (%B) = value_param call_param0
// CHECK:STDOUT:       %.loc6_16.1: type = splice_block %Self.ref.loc6 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:         %.loc6_16.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:         %Self.ref.loc6: type = name_ref Self, %.loc6_16.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       }
// CHECK:STDOUT:       %self.loc6: @B.F.%B (%B) = bind_name self, %self.param.loc6
// CHECK:STDOUT:       %a.param.loc6: @B.F.%T.loc6 (%T) = value_param call_param1
// CHECK:STDOUT:       %T.ref.loc6: type = name_ref T, @A.%T.loc4_9.2 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:       %a.loc6: @B.F.%T.loc6 (%T) = bind_name a, %a.param.loc6
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%B
// CHECK:STDOUT:     .T = <poisoned>
// CHECK:STDOUT:     .F = %B.F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @B.F(@A.%T.loc4_9.2: type, @B.%N.loc5_11.2: @B.%T (%T)) {
// CHECK:STDOUT:   %T.loc6: type = bind_symbolic_name T, 0 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:   %N.loc6: @B.F.%T.loc6 (%T) = bind_symbolic_name N, 1 [symbolic = %N.loc6 (constants.%N)]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T.loc6, %N.loc6) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:   %pattern_type.loc6_10: type = pattern_type %B [symbolic = %pattern_type.loc6_10 (constants.%pattern_type.13e)]
// CHECK:STDOUT:   %pattern_type.loc6_22: type = pattern_type %T.loc6 [symbolic = %pattern_type.loc6_22 (constants.%pattern_type.7dc)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc10_31: <witness> = require_complete_type %B [symbolic = %require_complete.loc10_31 (constants.%require_complete.fca)]
// CHECK:STDOUT:   %require_complete.loc10_40: <witness> = require_complete_type %T.loc6 [symbolic = %require_complete.loc10_40 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc10: @B.F.%B (%B), %a.param.loc10: @B.F.%T.loc6 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @A(constants.%T) {
// CHECK:STDOUT:   %T.loc4_9.1 => constants.%T
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.type => constants.%B.type
// CHECK:STDOUT:   %B.generic => constants.%B.generic
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT:   %N.loc5_11.1 => constants.%N
// CHECK:STDOUT:   %pattern_type => constants.%pattern_type.7dc
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %B.F.type => constants.%B.F.type
// CHECK:STDOUT:   %B.F => constants.%B.F
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @B.F(constants.%T, constants.%N) {
// CHECK:STDOUT:   %T.loc6 => constants.%T
// CHECK:STDOUT:   %N.loc6 => constants.%N
// CHECK:STDOUT:   %B => constants.%B
// CHECK:STDOUT:   %pattern_type.loc6_10 => constants.%pattern_type.13e
// CHECK:STDOUT:   %pattern_type.loc6_22 => constants.%pattern_type.7dc
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_not_generic_vs_generic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %NotGeneric: type = class_type @NotGeneric [concrete]
// CHECK:STDOUT:   %NotGeneric.F.type: type = fn_type @NotGeneric.F [concrete]
// CHECK:STDOUT:   %NotGeneric.F: %NotGeneric.F.type = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
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
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {
// CHECK:STDOUT:     %T.loc15_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @NotGeneric {
// CHECK:STDOUT:   %NotGeneric.F.decl: %NotGeneric.F.type = fn_decl @NotGeneric.F [concrete = constants.%NotGeneric.F] {} {}
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:   complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%NotGeneric
// CHECK:STDOUT:   .F = %NotGeneric.F.decl
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @NotGeneric.F();
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(%T.loc15_15.2: type) {
// CHECK:STDOUT:   %T.loc15_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   fn() {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @F(constants.%T) {
// CHECK:STDOUT:   %T.loc15_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_few_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.TooFew.type: type = fn_type @Generic.TooFew, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.TooFew: %Generic.TooFew.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %TooFew.type: type = fn_type @TooFew [concrete]
// CHECK:STDOUT:   %TooFew: %TooFew.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooFew.decl: %TooFew.type = fn_decl @TooFew [concrete = constants.%TooFew] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.TooFew.type: type = fn_type @Generic.TooFew, @Generic(%T.loc4_15.1) [symbolic = %Generic.TooFew.type (constants.%Generic.TooFew.type)]
// CHECK:STDOUT:   %Generic.TooFew: @Generic.%Generic.TooFew.type (%Generic.TooFew.type) = struct_value () [symbolic = %Generic.TooFew (constants.%Generic.TooFew)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Generic.TooFew.decl: @Generic.%Generic.TooFew.type (%Generic.TooFew.type) = fn_decl @Generic.TooFew [symbolic = @Generic.%Generic.TooFew (constants.%Generic.TooFew)] {} {}
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .TooFew = %Generic.TooFew.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.TooFew(@Generic.%T.loc4_15.2: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @TooFew() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic(constants.%T) {
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.TooFew(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_too_many_args.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.TooMany.type: type = fn_type @Generic.TooMany, @Generic(%T) [symbolic]
// CHECK:STDOUT:   %Generic.TooMany: %Generic.TooMany.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %U: type = bind_symbolic_name U, 1 [symbolic]
// CHECK:STDOUT:   %TooMany.type: type = fn_type @TooMany [concrete]
// CHECK:STDOUT:   %TooMany: %TooMany.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %TooMany.decl: %TooMany.type = fn_decl @TooMany [concrete = constants.%TooMany] {} {
// CHECK:STDOUT:     %T.loc15_12.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T)]
// CHECK:STDOUT:     %U.loc15_22.2: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_22.1 (constants.%U)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.TooMany.type: type = fn_type @Generic.TooMany, @Generic(%T.loc4_15.1) [symbolic = %Generic.TooMany.type (constants.%Generic.TooMany.type)]
// CHECK:STDOUT:   %Generic.TooMany: @Generic.%Generic.TooMany.type (%Generic.TooMany.type) = struct_value () [symbolic = %Generic.TooMany (constants.%Generic.TooMany)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Generic.TooMany.decl: @Generic.%Generic.TooMany.type (%Generic.TooMany.type) = fn_decl @Generic.TooMany [symbolic = @Generic.%Generic.TooMany (constants.%Generic.TooMany)] {} {}
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .TooMany = %Generic.TooMany.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.TooMany(@Generic.%T.loc4_15.2: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @TooMany(%T.loc15_12.2: type, %U.loc15_22.2: type) {
// CHECK:STDOUT:   %T.loc15_12.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T)]
// CHECK:STDOUT:   %U.loc15_22.1: type = bind_symbolic_name U, 1 [symbolic = %U.loc15_22.1 (constants.%U)]
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
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.TooMany(constants.%T) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @TooMany(constants.%T, constants.%U) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%T
// CHECK:STDOUT:   %U.loc15_22.1 => constants.%U
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_mismatched_wrong_arg_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T.8b3: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %pattern_type.98f: type = pattern_type type [concrete]
// CHECK:STDOUT:   %Generic.type: type = generic_class_type @Generic [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %Generic.generic: %Generic.type = struct_value () [concrete]
// CHECK:STDOUT:   %Generic: type = class_type @Generic, @Generic(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Generic.WrongType.type: type = fn_type @Generic.WrongType, @Generic(%T.8b3) [symbolic]
// CHECK:STDOUT:   %Generic.WrongType: %Generic.WrongType.type = struct_value () [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %T.7a6: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %WrongType.type: type = fn_type @WrongType [concrete]
// CHECK:STDOUT:   %WrongType: %WrongType.type = struct_value () [concrete]
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
// CHECK:STDOUT:     %T.patt: %pattern_type.98f = symbolic_binding_pattern T, 0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc4_15.2: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T.8b3)]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %WrongType.decl: %WrongType.type = fn_decl @WrongType [concrete = constants.%WrongType] {} {
// CHECK:STDOUT:     %.loc15_17.1: type = splice_block %.loc15_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc15_17.2: %empty_tuple.type = tuple_literal ()
// CHECK:STDOUT:       %.loc15_17.3: type = converted %.loc15_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %T.loc15_12.2: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T.7a6)]
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @Generic(%T.loc4_15.2: type) {
// CHECK:STDOUT:   %T.loc4_15.1: type = bind_symbolic_name T, 0 [symbolic = %T.loc4_15.1 (constants.%T.8b3)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %Generic.WrongType.type: type = fn_type @Generic.WrongType, @Generic(%T.loc4_15.1) [symbolic = %Generic.WrongType.type (constants.%Generic.WrongType.type)]
// CHECK:STDOUT:   %Generic.WrongType: @Generic.%Generic.WrongType.type (%Generic.WrongType.type) = struct_value () [symbolic = %Generic.WrongType (constants.%Generic.WrongType)]
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %Generic.WrongType.decl: @Generic.%Generic.WrongType.type (%Generic.WrongType.type) = fn_decl @Generic.WrongType [symbolic = @Generic.%Generic.WrongType (constants.%Generic.WrongType)] {} {}
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%Generic
// CHECK:STDOUT:     .WrongType = %Generic.WrongType.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @Generic.WrongType(@Generic.%T.loc4_15.2: type) {
// CHECK:STDOUT:   fn();
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @WrongType(%T.loc15_12.2: %empty_tuple.type) {
// CHECK:STDOUT:   %T.loc15_12.1: %empty_tuple.type = bind_symbolic_name T, 0 [symbolic = %T.loc15_12.1 (constants.%T.7a6)]
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
// CHECK:STDOUT:   %T.loc4_15.1 => constants.%T.8b3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: specific @Generic.WrongType(constants.%T.8b3) {}
// CHECK:STDOUT:
// CHECK:STDOUT: specific @WrongType(constants.%T.7a6) {
// CHECK:STDOUT:   %T.loc15_12.1 => constants.%T.7a6
// CHECK:STDOUT: }
// CHECK:STDOUT:
