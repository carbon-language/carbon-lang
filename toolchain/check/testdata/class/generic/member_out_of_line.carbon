// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
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
  //@dump-sem-ir-begin
  return n;
  //@dump-sem-ir-end
}

fn Class(T:! type).G[self: Self]() -> T {
  //@dump-sem-ir-begin
  return self.n;
  //@dump-sem-ir-end
}

// --- nested.carbon

library "[[@TEST_NAME]]";

class A(T:! type) {
  class B(N:! T) {
    fn F[self: Self](a: T);
  }
}

//@dump-sem-ir-begin
fn A(T:! type).B(N:! T).F[self: Self](a: T) {}
//@dump-sem-ir-end

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
// CHECK:STDOUT:   %Class: type = class_type @Class, @Class(%T) [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%n.param.loc10: @F.%T.loc5 (%T)) -> @F.%T.loc5 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %n.ref: @F.%T.loc5 (%T) = name_ref n, %n.loc10
// CHECK:STDOUT:     return %n.ref
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @G(@Class.%T.loc4_13.1: type) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:   %Class.elem: type = unbound_element_type %Class, %T.loc6 [symbolic = %Class.elem (constants.%Class.elem)]
// CHECK:STDOUT:   %require_complete.loc18: <witness> = require_complete_type %T.loc6 [symbolic = %require_complete.loc18 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc16: @G.%Class (%Class)) -> @G.%T.loc6 (%T) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     %self.ref: @G.%Class (%Class) = name_ref self, %self.loc16
// CHECK:STDOUT:     %n.ref: @G.%Class.elem (%Class.elem) = name_ref n, @Class.%.loc7 [concrete = @Class.%.loc7]
// CHECK:STDOUT:     %.loc18_14.1: ref @G.%T.loc6 (%T) = class_element_access %self.ref, element0
// CHECK:STDOUT:     %.loc18_14.2: @G.%T.loc6 (%T) = bind_value %.loc18_14.1
// CHECK:STDOUT:     return %.loc18_14.2
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = bind_symbolic_name T, 0 [symbolic]
// CHECK:STDOUT:   %N: %T = bind_symbolic_name N, 1 [symbolic]
// CHECK:STDOUT:   %pattern_type.7dc: type = pattern_type %T [symbolic]
// CHECK:STDOUT:   %B: type = class_type @B, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %pattern_type.13e: type = pattern_type %B [symbolic]
// CHECK:STDOUT:   %F.type: type = fn_type @F, @B(%T, %N) [symbolic]
// CHECK:STDOUT:   %F: %F.type = struct_value () [symbolic]
// CHECK:STDOUT:   %require_complete.fca: <witness> = require_complete_type %B [symbolic]
// CHECK:STDOUT:   %require_complete.4ae: <witness> = require_complete_type %T [symbolic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [symbolic = constants.%F] {
// CHECK:STDOUT:     %self.patt: @F.%pattern_type.loc6_10 (%pattern_type.13e) = binding_pattern self [concrete]
// CHECK:STDOUT:     %self.param_patt: @F.%pattern_type.loc6_10 (%pattern_type.13e) = value_param_pattern %self.patt, call_param0 [concrete]
// CHECK:STDOUT:     %a.patt: @F.%pattern_type.loc6_22 (%pattern_type.7dc) = binding_pattern a [concrete]
// CHECK:STDOUT:     %a.param_patt: @F.%pattern_type.loc6_22 (%pattern_type.7dc) = value_param_pattern %a.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %T.loc11: type = bind_symbolic_name T, 0 [symbolic = @A.%T.loc4_9.2 (constants.%T)]
// CHECK:STDOUT:     %T.ref.loc11_22: type = name_ref T, %T.loc11 [symbolic = @B.%T (constants.%T)]
// CHECK:STDOUT:     %N.loc11: @B.%T (%T) = bind_symbolic_name N, 1 [symbolic = @B.%N.loc5_11.2 (constants.%N)]
// CHECK:STDOUT:     %self.param.loc11: @F.%B (%B) = value_param call_param0
// CHECK:STDOUT:     %.loc11_33.1: type = splice_block %Self.ref.loc11 [symbolic = %B (constants.%B)] {
// CHECK:STDOUT:       %.loc11_33.2: type = specific_constant constants.%B, @B(constants.%T, constants.%N) [symbolic = %B (constants.%B)]
// CHECK:STDOUT:       %Self.ref.loc11: type = name_ref Self, %.loc11_33.2 [symbolic = %B (constants.%B)]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %self.loc11: @F.%B (%B) = bind_name self, %self.param.loc11
// CHECK:STDOUT:     %a.param.loc11: @F.%T.loc6 (%T) = value_param call_param1
// CHECK:STDOUT:     %T.ref.loc11_42: type = name_ref T, %T.loc11 [symbolic = %T.loc6 (constants.%T)]
// CHECK:STDOUT:     %a.loc11: @F.%T.loc6 (%T) = bind_name a, %a.param.loc11
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic fn @F(@A.%T.loc4_9.1: type, @B.%N.loc5_11.1: @B.%T (%T)) {
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:   %require_complete.loc11_31: <witness> = require_complete_type %B [symbolic = %require_complete.loc11_31 (constants.%require_complete.fca)]
// CHECK:STDOUT:   %require_complete.loc11_40: <witness> = require_complete_type %T.loc6 [symbolic = %require_complete.loc11_40 (constants.%require_complete.4ae)]
// CHECK:STDOUT:
// CHECK:STDOUT:   fn(%self.param.loc11: @F.%B (%B), %a.param.loc11: @F.%T.loc6 (%T)) {
// CHECK:STDOUT:   !entry:
// CHECK:STDOUT:     return
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
