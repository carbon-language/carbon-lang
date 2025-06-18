// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/convert.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/class/partial.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/class/partial.carbon

// --- basic.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- basic_abstract.carbon
library "[[@TEST_NAME]]";

abstract class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- todo_fail_partial_nonclass.carbon
library "[[@TEST_NAME]]";

class C { }

fn G(p: partial (C*));

// --- todo_fail_partial_pointer.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
fn B(p: partial (C*));
//@dump-sem-ir-end

// --- fail_todo_collapse.carbon
library "[[@TEST_NAME]]";

base class C { }

// `partial (partial C)` should be the same type as `partial C`.
//@dump-sem-ir-begin
fn F(p: partial C*) -> partial (partial C*) {
  // CHECK:STDERR: fail_todo_collapse.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `partial C*` to `partial partial C*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_todo_collapse.carbon:[[@LINE+4]]:3: note: type `partial C*` does not implement interface `Core.ImplicitAs(partial partial C*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}
//@dump-sem-ir-end

// --- fail_convert_to_nonpartial.carbon
library "[[@TEST_NAME]]";

base class C { }

fn G(p: partial C*) -> C* {
  // CHECK:STDERR: fail_convert_to_nonpartial.carbon:[[@LINE+7]]:3: error: cannot implicitly convert expression of type `partial C*` to `C*` [ConversionFailure]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR: fail_convert_to_nonpartial.carbon:[[@LINE+4]]:3: note: type `partial C*` does not implement interface `Core.ImplicitAs(C*)` [MissingImplInMemberAccessNote]
  // CHECK:STDERR:   return p;
  // CHECK:STDERR:   ^~~~~~~~~
  // CHECK:STDERR:
  return p;
}

// CHECK:STDOUT: --- basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- basic_abstract.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %A.type: type = fn_type @A [concrete]
// CHECK:STDOUT:   %A: %A.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %A.decl: %A.type = fn_decl @A [concrete = constants.%A] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.43f = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.43f] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.43f = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @A(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_pointer.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %C [concrete]
// CHECK:STDOUT:   %.21c: type = partial_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.21c [concrete]
// CHECK:STDOUT:   %B.type: type = fn_type @B [concrete]
// CHECK:STDOUT:   %B: %B.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %B.decl: %B.type = fn_decl @B [concrete = constants.%B] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.21c = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.21c] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %ptr: type = ptr_type %C.ref [concrete = constants.%ptr]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %ptr [concrete = constants.%.21c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.21c = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @B(%p.param: %.21c);
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_collapse.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %ptr: type = ptr_type %.43f [concrete]
// CHECK:STDOUT:   %pattern_type.a4d: type = pattern_type %ptr [concrete]
// CHECK:STDOUT:   %.ba3: type = partial_type %ptr [concrete]
// CHECK:STDOUT:   %pattern_type.e22: type = pattern_type %.ba3 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type.a4d = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type.a4d = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:     %return.patt: %pattern_type.e22 = return_slot_pattern [concrete]
// CHECK:STDOUT:     %return.param_patt: %pattern_type.e22 = out_param_pattern %return.patt, call_param1 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %C.ref.loc7_41: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:     %.loc7_33: type = partial_type %C.ref.loc7_41 [concrete = constants.%.43f]
// CHECK:STDOUT:     %ptr.loc7_42: type = ptr_type %.loc7_33 [concrete = constants.%ptr]
// CHECK:STDOUT:     %.loc7_24: type = partial_type %ptr.loc7_42 [concrete = constants.%.ba3]
// CHECK:STDOUT:     %p.param: %ptr = value_param call_param0
// CHECK:STDOUT:     %.loc7_18: type = splice_block %ptr.loc7_18 [concrete = constants.%ptr] {
// CHECK:STDOUT:       %C.ref.loc7_17: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc7_9: type = partial_type %C.ref.loc7_17 [concrete = constants.%.43f]
// CHECK:STDOUT:       %ptr.loc7_18: type = ptr_type %.loc7_9 [concrete = constants.%ptr]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %ptr = bind_name p, %p.param
// CHECK:STDOUT:     %return.param: ref %.ba3 = out_param call_param1
// CHECK:STDOUT:     %return: ref %.ba3 = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %ptr) -> %.ba3 {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %p.ref: %ptr = name_ref p, %p
// CHECK:STDOUT:   %.loc15: %.ba3 = converted %p.ref, <error> [concrete = <error>]
// CHECK:STDOUT:   return <error>
// CHECK:STDOUT: }
// CHECK:STDOUT:
