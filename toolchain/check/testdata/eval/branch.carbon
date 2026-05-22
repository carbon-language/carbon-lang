// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/bool.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/eval/branch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/eval/branch.carbon

// --- types.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

fn MakeA() -> A { return {}; }
fn MakeB() -> B { return {}; }

// --- if_statement.carbon

library "[[@TEST_NAME]]";
import library "types";

eval fn F(b: bool) -> type {
  if (b) {
    return A;
  } else {
    return B;
  }
}

//@dump-sem-ir-begin
var a: F(true) = MakeA();
var b: F(false) = MakeB();
//@dump-sem-ir-end

// --- if_expression_in_eval_fn.carbon

library "[[@TEST_NAME]]";
import library "types";

eval fn F(b: bool) -> type {
  return if b then A else B;
}

//@dump-sem-ir-begin
var a: F(true) = MakeA();
var b: F(false) = MakeB();
//@dump-sem-ir-end

// --- fail_todo_if_expression_direct.carbon

library "[[@TEST_NAME]]";
import library "types";

// CHECK:STDERR: fail_todo_if_expression_direct.carbon:[[@LINE+4]]:8: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var a: if true then A else B = MakeA();
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: if true then A else B = MakeA();
var b: if false then A else B = MakeB();

// CHECK:STDOUT: --- if_statement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.29a: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %MakeA.type: type = fn_type @MakeA [concrete]
// CHECK:STDOUT:   %MakeA: %MakeA.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type.82f: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.MakeA: %MakeA.type = import_ref Main//types, MakeA, loaded [concrete = constants.%MakeA]
// CHECK:STDOUT:   %Main.MakeB: %MakeB.type = import_ref Main//types, MakeB, loaded [concrete = constants.%MakeB]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.29a = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.29a = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc14_14.1: type = splice_block %.loc14_14.3 [concrete = constants.%A] {
// CHECK:STDOUT:     %F.ref.loc14: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %F.call.loc14: init type = call %F.ref.loc14(%true) [concrete = constants.%A]
// CHECK:STDOUT:     %.loc14_14.2: type = value_of_initializer %F.call.loc14 [concrete = constants.%A]
// CHECK:STDOUT:     %.loc14_14.3: type = converted %F.call.loc14, %.loc14_14.2 [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.82f = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.82f = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc15_15.1: type = splice_block %.loc15_15.3 [concrete = constants.%B] {
// CHECK:STDOUT:     %F.ref.loc15: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %F.call.loc15: init type = call %F.ref.loc15(%false) [concrete = constants.%B]
// CHECK:STDOUT:     %.loc15_15.2: type = value_of_initializer %F.call.loc15 [concrete = constants.%B]
// CHECK:STDOUT:     %.loc15_15.3: type = converted %F.call.loc15, %.loc15_15.2 [concrete = constants.%B]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %B = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeA.ref: %MakeA.type = name_ref MakeA, imports.%Main.MakeA [concrete = constants.%MakeA]
// CHECK:STDOUT:   %.loc14: ref %A = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %MakeA.call: init %A to %.loc14 = call %MakeA.ref()
// CHECK:STDOUT:   assign file.%a.var, %MakeA.call
// CHECK:STDOUT:   %MakeB.ref: %MakeB.type = name_ref MakeB, imports.%Main.MakeB [concrete = constants.%MakeB]
// CHECK:STDOUT:   %.loc15: ref %B = splice_block file.%b.var [concrete = file.%b.var] {}
// CHECK:STDOUT:   %MakeB.call: init %B to %.loc15 = call %MakeB.ref()
// CHECK:STDOUT:   assign file.%b.var, %MakeB.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- if_expression_in_eval_fn.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %pattern_type.fe8: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %MakeA.type: type = fn_type @MakeA [concrete]
// CHECK:STDOUT:   %MakeA: %MakeA.type = struct_value () [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type.61d: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Main.MakeA: %MakeA.type = import_ref Main//types, MakeA, loaded [concrete = constants.%MakeA]
// CHECK:STDOUT:   %Main.MakeB: %MakeB.type = import_ref Main//types, MakeB, loaded [concrete = constants.%MakeB]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.fe8 = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.fe8 = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc10_14.1: type = splice_block %.loc10_14.3 [concrete = constants.%A] {
// CHECK:STDOUT:     %F.ref.loc10: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %F.call.loc10: init type = call %F.ref.loc10(%true) [concrete = constants.%A]
// CHECK:STDOUT:     %.loc10_14.2: type = value_of_initializer %F.call.loc10 [concrete = constants.%A]
// CHECK:STDOUT:     %.loc10_14.3: type = converted %F.call.loc10, %.loc10_14.2 [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.61d = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.61d = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc11_15.1: type = splice_block %.loc11_15.3 [concrete = constants.%B] {
// CHECK:STDOUT:     %F.ref.loc11: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %F.call.loc11: init type = call %F.ref.loc11(%false) [concrete = constants.%B]
// CHECK:STDOUT:     %.loc11_15.2: type = value_of_initializer %F.call.loc11 [concrete = constants.%B]
// CHECK:STDOUT:     %.loc11_15.3: type = converted %F.call.loc11, %.loc11_15.2 [concrete = constants.%B]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %B = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeA.ref: %MakeA.type = name_ref MakeA, imports.%Main.MakeA [concrete = constants.%MakeA]
// CHECK:STDOUT:   %.loc10: ref %A = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %MakeA.call: init %A to %.loc10 = call %MakeA.ref()
// CHECK:STDOUT:   assign file.%a.var, %MakeA.call
// CHECK:STDOUT:   %MakeB.ref: %MakeB.type = name_ref MakeB, imports.%Main.MakeB [concrete = constants.%MakeB]
// CHECK:STDOUT:   %.loc11: ref %B = splice_block file.%b.var [concrete = file.%b.var] {}
// CHECK:STDOUT:   %MakeB.call: init %B to %.loc11 = call %MakeB.ref()
// CHECK:STDOUT:   assign file.%b.var, %MakeB.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
