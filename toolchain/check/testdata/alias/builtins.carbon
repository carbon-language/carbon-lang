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

// --- fail_i32.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_i32.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias a = i32;
// CHECK:STDERR:           ^~~
// CHECK:STDERR:
alias a = i32;
//@dump-sem-ir-end

// --- fail_bool.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_bool.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias b = bool;
// CHECK:STDERR:           ^~~~
// CHECK:STDERR:
alias b = bool;
//@dump-sem-ir-end

// --- fail_bool_value.carbon

library "[[@TEST_NAME]]";

//@dump-sem-ir-begin
// CHECK:STDERR: fail_bool_value.carbon:[[@LINE+4]]:11: error: alias initializer must be a name reference [AliasRequiresNameRef]
// CHECK:STDERR: alias a = false;
// CHECK:STDERR:           ^~~~~
// CHECK:STDERR:
alias a = false;

// This fails silently due to the above diagnostic.
let a_test: bool = a;
//@dump-sem-ir-end

// CHECK:STDOUT: --- fail_i32.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %a: <error> = bind_alias a, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bool.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:   %b: <error> = bind_alias b, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_bool_value.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete]
// CHECK:STDOUT:   %Bool.type: type = fn_type @Bool [concrete]
// CHECK:STDOUT:   %Bool: %Bool.type = struct_value () [concrete]
// CHECK:STDOUT:   %pattern_type.831: type = pattern_type bool [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   %false: bool = bool_literal false [concrete = constants.%false]
// CHECK:STDOUT:   %a: <error> = bind_alias a, <error> [concrete = <error>]
// CHECK:STDOUT:   name_binding_decl {
// CHECK:STDOUT:     %a_test.patt: %pattern_type.831 = binding_pattern a_test [concrete]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.loc12_13.1: type = splice_block %.loc12_13.3 [concrete = bool] {
// CHECK:STDOUT:     %Bool.call: init type = call constants.%Bool() [concrete = bool]
// CHECK:STDOUT:     %.loc12_13.2: type = value_of_initializer %Bool.call [concrete = bool]
// CHECK:STDOUT:     %.loc12_13.3: type = converted %Bool.call, %.loc12_13.2 [concrete = bool]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %a_test: bool = bind_name a_test, <error> [concrete = <error>]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @__global_init() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %a.ref: <error> = name_ref a, file.%a [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
