// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/if_expr/fail_not_in_function.carbon

// --- fail_basic.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_basic.carbon:[[@LINE+12]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_basic.carbon:[[@LINE+8]]:22: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:                      ^~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_basic.carbon:[[@LINE+4]]:14: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: let x: i32 = if true then 1 else 0;
// CHECK:STDERR:              ^~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
let x: i32 = if true then 1 else 0;

// --- fail_types.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_types.carbon:[[@LINE+4]]:8: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: var y: if true then i32 else f64;
// CHECK:STDERR:        ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
var y: if true then i32 else f64;

// --- fail_in_param.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_in_param.carbon:[[@LINE+4]]:9: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: fn F(a: if true then i32 else f64);
// CHECK:STDERR:         ^~~~~~~~~~~~~~~~~~~~~~~~~
// CHECK:STDERR:
fn F(a: if true then i32 else f64);

// --- fail_class.carbon

library "[[@TEST_NAME]]";

class C {
  // CHECK:STDERR: fail_class.carbon:[[@LINE+12]]:10: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+8]]:18: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:                  ^~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: fail_class.carbon:[[@LINE+4]]:10: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
  // CHECK:STDERR:   var n: if true then i32 else f64;
  // CHECK:STDERR:          ^~~~~~~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  var n: if true then i32 else f64;
}

// --- fail_nested_class.carbon

library "[[@TEST_NAME]]";

base class C(T:! type) {}

fn F() {
  class B {
    // CHECK:STDERR: fail_nested_class.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
    // CHECK:STDERR:     extend base: C(if true then i32 else i32);
    // CHECK:STDERR:                    ^~~~~~~~~~~~~~~~~~~~~~~~~
    // CHECK:STDERR:
    extend base: C(if true then i32 else i32);
  }
}

// CHECK:STDOUT: --- fail_basic.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_types.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_in_param.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %C: type = class_type @C [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @C {
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%C
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested_class.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%T) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type.357: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(<unexpected>.inst68000015.loc4_14: type) {
// CHECK:STDOUT:   %T: type = symbolic_binding T, 0 [symbolic = %T (constants.%T)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness constants.%empty_struct_type [concrete = constants.%complete_type.357]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, <unexpected>.inst6800001A.loc4_24 [concrete = constants.%C.generic]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %true br !if.expr.then else br !if.expr.else
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%T) {
// CHECK:STDOUT:   %T => constants.%T
// CHECK:STDOUT: }
// CHECK:STDOUT:
