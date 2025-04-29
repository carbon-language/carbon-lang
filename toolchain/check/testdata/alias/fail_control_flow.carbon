// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/fail_control_flow.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/fail_control_flow.carbon

// --- fail_simple.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_simple.carbon:[[@LINE+8]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~
// CHECK:STDERR:
// CHECK:STDERR: fail_simple.carbon:[[@LINE+4]]:11: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
// CHECK:STDERR: alias a = true or false;
// CHECK:STDERR:           ^~~~~~~~~~~~~
// CHECK:STDERR:
alias a = true or false;

// --- fail_nested.carbon

library "[[@TEST_NAME]]";

base class C(B:! bool) {}

fn F() {
  class B {
    // CHECK:STDERR: fail_nested.carbon:[[@LINE+4]]:20: error: semantics TODO: `Control flow expressions are currently only supported inside functions.` [SemanticsTodo]
    // CHECK:STDERR:     extend base: C(true or false);
    // CHECK:STDERR:                    ^~~~~~~~~~~~~
    // CHECK:STDERR:
    extend base: C(true or false);
  }
}

// CHECK:STDOUT: --- fail_simple.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_nested.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %B.7dd: bool = bind_symbolic_name B, 0 [symbolic]
// CHECK:STDOUT:   %C.type: type = generic_class_type @C [concrete]
// CHECK:STDOUT:   %C.generic: %C.type = struct_value () [concrete]
// CHECK:STDOUT:   %C: type = class_type @C, @C(%B.7dd) [symbolic]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete]
// CHECK:STDOUT:   %B.d97: type = class_type @B [concrete]
// CHECK:STDOUT:   %true: bool = bool_literal true [concrete]
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {}
// CHECK:STDOUT:
// CHECK:STDOUT: generic class @C(<unexpected>.inst28.loc4_14: bool) {
// CHECK:STDOUT:   %B: bool = bind_symbolic_name B, 0 [symbolic = %B (constants.%B.7dd)]
// CHECK:STDOUT:
// CHECK:STDOUT: !definition:
// CHECK:STDOUT:
// CHECK:STDOUT:   class {
// CHECK:STDOUT:     %empty_struct_type: type = struct_type {} [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %complete_type: <witness> = complete_type_witness %empty_struct_type [concrete = constants.%complete_type]
// CHECK:STDOUT:     complete_type_witness = %complete_type
// CHECK:STDOUT:
// CHECK:STDOUT:   !members:
// CHECK:STDOUT:     .Self = constants.%C
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: class @B {
// CHECK:STDOUT:   %C.ref: %C.type = name_ref C, <unexpected>.inst33.loc4_24 [concrete = constants.%C.generic]
// CHECK:STDOUT:   %true.loc12_20: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   %.loc12: bool = not %true.loc12_20 [concrete = constants.%false]
// CHECK:STDOUT:   %true.loc12_25: bool = bool_literal true [concrete = constants.%true]
// CHECK:STDOUT:   if %.loc12 br !or.rhs else br !or.result(%true.loc12_25)
// CHECK:STDOUT:   complete_type_witness = invalid
// CHECK:STDOUT:
// CHECK:STDOUT: !members:
// CHECK:STDOUT:   .Self = constants.%B.d97
// CHECK:STDOUT:   .C = <poisoned>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F();
// CHECK:STDOUT:
// CHECK:STDOUT: specific @C(constants.%B.7dd) {
// CHECK:STDOUT:   %B => constants.%B.7dd
// CHECK:STDOUT: }
// CHECK:STDOUT:
