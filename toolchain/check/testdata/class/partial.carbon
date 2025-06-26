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

// --- base.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- abstract.carbon
library "[[@TEST_NAME]]";

abstract class C { }

//@dump-sem-ir-begin
fn A(p: partial C);
//@dump-sem-ir-end

// --- todo_fail_partial_nondynamic.carbon
library "[[@TEST_NAME]]";

class C { }

//@dump-sem-ir-begin
fn G(p: partial C);
//@dump-sem-ir-end

// --- todo_fail_partial_final.carbon
library "[[@TEST_NAME]]";

base class Base { }
class Derived {
  extend base: Base;
}

//@dump-sem-ir-begin
fn G(p: partial Derived);
//@dump-sem-ir-end

// --- todo_fail_partial_decl.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
fn G(p: partial C);
//@dump-sem-ir-end

// --- todo_fail_partial_tuple.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
fn G(p: partial (C, C));
//@dump-sem-ir-end

// --- todo_fail_partial_struct.carbon
library "[[@TEST_NAME]]";

class C;

//@dump-sem-ir-begin
fn G(p: partial {.x: C});
//@dump-sem-ir-end

// --- todo_fail_duplicate.carbon
library "[[@TEST_NAME]]";

base class C { }

//@dump-sem-ir-begin
fn F(p: partial (partial C));
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

// CHECK:STDOUT: --- base.carbon
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
// CHECK:STDOUT: --- abstract.carbon
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
// CHECK:STDOUT: --- todo_fail_partial_nondynamic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT: fn @G(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_final.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Derived: type = class_type @Derived [concrete]
// CHECK:STDOUT:   %.604: type = partial_type %Derived [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.604 [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.604 = value_param call_param0
// CHECK:STDOUT:     %.loc9_9.1: type = splice_block %.loc9_9.2 [concrete = constants.%.604] {
// CHECK:STDOUT:       %Derived.ref: type = name_ref Derived, file.%Derived.decl [concrete = constants.%Derived]
// CHECK:STDOUT:       %.loc9_9.2: type = partial_type %Derived.ref [concrete = constants.%.604]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.604 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.604);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_decl.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.43f [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
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
// CHECK:STDOUT: fn @G(%p.param: %.43f);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_tuple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %tuple.type.24b: type = tuple_type (type, type) [concrete]
// CHECK:STDOUT:   %tuple.type.56b: type = tuple_type (%C, %C) [concrete]
// CHECK:STDOUT:   %.62c: type = partial_type %tuple.type.56b [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.62c [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.62c = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.3 [concrete = constants.%.62c] {
// CHECK:STDOUT:       %C.ref.loc6_18: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %C.ref.loc6_21: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_22: %tuple.type.24b = tuple_literal (%C.ref.loc6_18, %C.ref.loc6_21)
// CHECK:STDOUT:       %.loc6_9.2: type = converted %.loc6_22, constants.%tuple.type.56b [concrete = constants.%tuple.type.56b]
// CHECK:STDOUT:       %.loc6_9.3: type = partial_type %.loc6_9.2 [concrete = constants.%.62c]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.62c = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.62c);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_partial_struct.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %struct_type.x: type = struct_type {.x: %C} [concrete]
// CHECK:STDOUT:   %.1ed: type = partial_type %struct_type.x [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.1ed [concrete]
// CHECK:STDOUT:   %G.type: type = fn_type @G [concrete]
// CHECK:STDOUT:   %G: %G.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %G.decl: %G.type = fn_decl @G [concrete = constants.%G] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.1ed = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.1ed] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %struct_type.x: type = struct_type {.x: %C} [concrete = constants.%struct_type.x]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %struct_type.x [concrete = constants.%.1ed]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.1ed = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G(%p.param: %.1ed);
// CHECK:STDOUT:
// CHECK:STDOUT: --- todo_fail_duplicate.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %.43f: type = partial_type %C [concrete]
// CHECK:STDOUT:   %.a34: type = partial_type %.43f [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type %.a34 [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {
// CHECK:STDOUT:     %p.patt: %pattern_type = binding_pattern p [concrete]
// CHECK:STDOUT:     %p.param_patt: %pattern_type = value_param_pattern %p.patt, call_param0 [concrete]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %p.param: %.a34 = value_param call_param0
// CHECK:STDOUT:     %.loc6_9.1: type = splice_block %.loc6_9.2 [concrete = constants.%.a34] {
// CHECK:STDOUT:       %C.ref: type = name_ref C, file.%C.decl [concrete = constants.%C]
// CHECK:STDOUT:       %.loc6_18: type = partial_type %C.ref [concrete = constants.%.43f]
// CHECK:STDOUT:       %.loc6_9.2: type = partial_type %.loc6_18 [concrete = constants.%.a34]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %p: %.a34 = bind_name p, %p.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F(%p.param: %.a34);
// CHECK:STDOUT:
