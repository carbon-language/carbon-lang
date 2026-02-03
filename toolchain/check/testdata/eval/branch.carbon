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

// --- if_statement.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

eval fn F(b: bool) -> type {
  if (b) {
    return A;
  } else {
    return B;
  }
}

fn MakeA() -> A { return {}; }
fn MakeB() -> B { return {}; }

//@dump-sem-ir-begin
var a: F(true) = MakeA();
var b: F(false) = MakeB();
//@dump-sem-ir-end

// --- fail_todo_if_expression_in_eval_fn.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

eval fn F(b: bool) -> type {
  // CHECK:STDERR: fail_todo_if_expression_in_eval_fn.carbon:[[@LINE+3]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
  // CHECK:STDERR:   return if b then A else B;
  // CHECK:STDERR:          ^~~~
  return if b then A else B;
}

// CHECK:STDERR: fail_todo_if_expression_in_eval_fn.carbon:[[@LINE+7]]:8: note: in call to F here [InCallToEvalFn]
// CHECK:STDERR: var a: F(true) = {} as A;
// CHECK:STDERR:        ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_todo_if_expression_in_eval_fn.carbon:[[@LINE-7]]:10: error: expression is runtime; expected constant [EvalRequiresConstantValue]
// CHECK:STDERR:   return if b then A else B;
// CHECK:STDERR:          ^~~~
var a: F(true) = {} as A;
// CHECK:STDERR: fail_todo_if_expression_in_eval_fn.carbon:[[@LINE+4]]:8: note: in call to F here [InCallToEvalFn]
// CHECK:STDERR: var b: F(false) = {} as B;
// CHECK:STDERR:        ^~~~~~~~
// CHECK:STDERR:
var b: F(false) = {} as B;

// --- fail_todo_if_expression_direct.carbon

library "[[@TEST_NAME]]";

class A {}
class B {}

// CHECK:STDERR: fail_todo_if_expression_direct.carbon:[[@LINE+4]]:8: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var a: if true then A else B = {} as A;
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var a: if true then A else B = {} as A;
var b: if false then A else B = {} as B;

// CHECK:STDOUT: --- if_statement.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %A: type = class_type @A [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1ab: type = pattern_type %A [concrete]
// CHECK:STDOUT:   %MakeA.type: type = fn_type @MakeA [concrete]
// CHECK:STDOUT:   %MakeA: %MakeA.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.1f4: type = pattern_type %B [concrete]
// CHECK:STDOUT:   %MakeB.type: type = fn_type @MakeB [concrete]
// CHECK:STDOUT:   %MakeB: %MakeB.type = struct_value () [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a.patt: %pattern_type.1ab = ref_binding_pattern a [concrete]
// CHECK:STDOUT:     %a.var_patt: %pattern_type.1ab = var_pattern %a.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a.var: ref %A = var %a.var_patt [concrete]
// CHECK:STDOUT:   %.loc19_14.1: type = splice_block %.loc19_14.3 [concrete = constants.%A] {
// CHECK:STDOUT:     %F.ref.loc19: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:     %F.call.loc19: init type = call %F.ref.loc19(%true) [concrete = constants.%A]
// CHECK:STDOUT:     %.loc19_14.2: type = value_of_initializer %F.call.loc19 [concrete = constants.%A]
// CHECK:STDOUT:     %.loc19_14.3: type = converted %F.call.loc19, %.loc19_14.2 [concrete = constants.%A]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a: ref %A = ref_binding a, %a.var [concrete = %a.var]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %b.patt: %pattern_type.1f4 = ref_binding_pattern b [concrete]
// CHECK:STDOUT:     %b.var_patt: %pattern_type.1f4 = var_pattern %b.patt [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b.var: ref %B = var %b.var_patt [concrete]
// CHECK:STDOUT:   %.loc20_15.1: type = splice_block %.loc20_15.3 [concrete = constants.%B] {
// CHECK:STDOUT:     %F.ref.loc20: %F.type = name_ref F, %F.decl [concrete = constants.%F]
// CHECK:STDOUT:     %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:     %F.call.loc20: init type = call %F.ref.loc20(%false) [concrete = constants.%B]
// CHECK:STDOUT:     %.loc20_15.2: type = value_of_initializer %F.call.loc20 [concrete = constants.%B]
// CHECK:STDOUT:     %.loc20_15.3: type = converted %F.call.loc20, %.loc20_15.2 [concrete = constants.%B]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %b: ref %B = ref_binding b, %b.var [concrete = %b.var]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %MakeA.ref: %MakeA.type = name_ref MakeA, file.%MakeA.decl [concrete = constants.%MakeA]
// CHECK:STDOUT:   %.loc19: ref %A = splice_block file.%a.var [concrete = file.%a.var] {}
// CHECK:STDOUT:   %MakeA.call: init %A to %.loc19 = call %MakeA.ref()
// CHECK:STDOUT:   assign file.%a.var, %MakeA.call
// CHECK:STDOUT:   %MakeB.ref: %MakeB.type = name_ref MakeB, file.%MakeB.decl [concrete = constants.%MakeB]
// CHECK:STDOUT:   %.loc20: ref %B = splice_block file.%b.var [concrete = file.%b.var] {}
// CHECK:STDOUT:   %MakeB.call: init %B to %.loc20 = call %MakeB.ref()
// CHECK:STDOUT:   assign file.%b.var, %MakeB.call
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
