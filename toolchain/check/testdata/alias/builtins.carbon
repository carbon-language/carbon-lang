// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/alias/builtins.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/alias/builtins.carbon

// --- i32.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
alias a = i32;
//@dump-sem-ir-end

// --- bool.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
alias b = bool;
//@dump-sem-ir-end

// --- bool_value.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
alias a = false;

let a_test: bool = a;
//@dump-sem-ir-end

// CHECK:STDOUT: --- i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %i32: type = type_literal constants.%i32 [concrete = constants.%i32]
// CHECK:STDOUT:   %a: type = alias_binding a, %i32 [concrete = constants.%i32]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %.loc5: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %b: type = alias_binding b, %.loc5 [concrete = bool]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- bool_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %pattern_type: type = pattern_type bool [concrete]
// CHECK:STDOUT:   %a_test.patt: %pattern_type = value_binding_pattern a_test [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %a: bool = alias_binding a, %false [concrete = constants.%false]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_test.patt: %pattern_type = value_binding_pattern a_test [concrete = constants.%a_test.patt]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc7: type = type_literal bool [concrete = bool]
// CHECK:STDOUT:   %a_test: bool = wrapper_binding a_test, @__global_init.%a.ref
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: bool = name_ref a, file.%a [concrete = constants.%false]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
