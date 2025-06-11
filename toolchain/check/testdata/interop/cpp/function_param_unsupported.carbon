// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// EXTRA-ARGS: --dump-sem-ir-ranges=only
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function_param_unsupported.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function_param_unsupported.carbon

// ============================================================================
// Unsupported primitive type
// ============================================================================

// --- unsupported_primitive_type.h

auto foo(float a) -> void;

// --- fail_todo_import_unsupported_primitive_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_primitive_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1.1);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(1.1);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1.1);
  //@dump-sem-ir-end
}

// ============================================================================
// Unsupported primitive type among params
// ============================================================================

// --- unsupported_primitive_type_among_params.h

auto foo(int a, float b) -> void;

// --- fail_todo_import_unsupported_primitive_type_among_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_primitive_type_among_params.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type_among_params.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1, 2.0);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type_among_params.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(1, 2.0);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1, 2.0);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_unsupported_primitive_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %float: f64 = float_literal 1.1000000000000001 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %float: f64 = float_literal 1.1000000000000001 [concrete = constants.%float]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_unsupported_primitive_type_among_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %float: f64 = float_literal 2 [concrete = constants.%float]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
