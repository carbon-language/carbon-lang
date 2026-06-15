// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/patterns/expression.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/patterns/expression.carbon

// --- fail_todo_let.carbon

library "[[@TEST_NAME]]";

// TODO: We should reject this for using a refutable pattern in an irrefutable pattern context.
let a: () = ();
// CHECK:STDERR: fail_todo_let.carbon:[[@LINE+4]]:5: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: let a = ();
// CHECK:STDERR:     ^
// CHECK:STDERR:
let a = ();

// --- fail_todo_in_tuple.carbon

library "[[@TEST_NAME]]";

// TODO: We should reject this for using a refutable pattern in an irrefutable pattern context.
let a: () = ();
// CHECK:STDERR: fail_todo_in_tuple.carbon:[[@LINE+4]]:13: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: let (b: (), a, c: ()) = ((), (), ());
// CHECK:STDERR:             ^
// CHECK:STDERR:
let (b: (), a, c: ()) = ((), (), ());

// --- fail_todo_in_function_param.carbon

library "[[@TEST_NAME]]";

// TODO: We should reject this for using a refutable pattern in an irrefutable pattern context.
let a: () = ();
// CHECK:STDERR: fail_todo_in_function_param.carbon:[[@LINE+4]]:6: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: fn F(a) {}
// CHECK:STDERR:      ^
// CHECK:STDERR:
fn F(a) {}

// --- fail_todo_parens.carbon

library "[[@TEST_NAME]]";

let a: () = ();
// TODO: Replace this with a valid expression pattern once we support refutable
// pattern contexts.
// Note that this is a parenthesized zero-tuple expression pattern, not a
// one-tuple pattern containing a zero-tuple.
// CHECK:STDERR: fail_todo_parens.carbon:[[@LINE+4]]:6: error: semantics TODO: `expression pattern` [SemanticsTodo]
// CHECK:STDERR: let (a) = ();
// CHECK:STDERR:      ^
// CHECK:STDERR:
let (a) = ();

// --- fail_todo_control_flow.carbon

library "[[@TEST_NAME]]";

fn F() {
  // TODO: Replace this with a valid expression pattern once we support refutable
  // pattern contexts.
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_control_flow.carbon:[[@LINE+8]]:8: error: semantics TODO: `expression pattern` [SemanticsTodo]
  // CHECK:STDERR:   let (if true then true else false, true and true) = (true, true);
  // CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_todo_control_flow.carbon:[[@LINE+4]]:38: error: semantics TODO: `expression pattern` [SemanticsTodo]
  // CHECK:STDERR:   let (if true then true else false, true and true) = (true, true);
  // CHECK:STDERR:                                      ^~~~~~~~~~~~~
  // CHECK:STDERR:
  let (if true then true else false, true and true) = (true, true);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_control_flow.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %tuple.type: type = tuple_type (bool, bool) [concrete]
// CHECK:STDOUT:   %pattern_type.860: type = pattern_type %tuple.type [concrete]
// CHECK:STDOUT:   %.2f0: %pattern_type.860 = tuple_pattern (@F.%expr_patt.loc16_26, @F.%expr_patt.loc16_43) [concrete]
// CHECK:STDOUT:   %tuple: %tuple.type = tuple_value (%true, %true) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %expr_patt.loc16_26: %pattern_type.831 = expr_pattern %.loc16_8 in [concrete] {
// CHECK:STDOUT:       %true.loc16_11: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:       if %true.loc16_11 br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:
// CHECK:STDOUT:     !if.expr.then:
// CHECK:STDOUT:       %true.loc16_21: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:       br !if.expr.result(%true.loc16_21)
// CHECK:STDOUT:
// CHECK:STDOUT:     !if.expr.else:
// CHECK:STDOUT:       %false.loc16_31: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:       br !if.expr.result(%false.loc16_31)
// CHECK:STDOUT:
// CHECK:STDOUT:     !if.expr.result:
// CHECK:STDOUT:       %.loc16_8: bool = block_arg !if.expr.result [concrete = constants.%true]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %expr_patt.loc16_43: %pattern_type.831 = expr_pattern %.loc16_43 in [concrete] {
// CHECK:STDOUT:       %true.loc16_38: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:       %false.loc16_43: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:       if %true.loc16_38 br !and.rhs else br !.loc16(%false.loc16_43)
// CHECK:STDOUT:
// CHECK:STDOUT:     !and.rhs:
// CHECK:STDOUT:       %true.loc16_47: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:       br !.loc16(%true.loc16_47)
// CHECK:STDOUT:
// CHECK:STDOUT:     !.loc16:
// CHECK:STDOUT:       %.loc16_43: bool = block_arg !.loc16 [concrete = constants.%true]
// CHECK:STDOUT:       <elided>
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %.loc16_51: %pattern_type.860 = tuple_pattern (%expr_patt.loc16_26, %expr_patt.loc16_43) [concrete = constants.%.2f0]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %true.loc16_56: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %true.loc16_62: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc16_66: %tuple.type = tuple_literal (%true.loc16_56, %true.loc16_62) [concrete = constants.%tuple]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
