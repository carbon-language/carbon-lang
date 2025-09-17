// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/param_unsupported.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/param_unsupported.carbon

// ============================================================================
// Unsupported primitive type
// ============================================================================

// --- unsupported_primitive_type.h

auto foo(_BitInt(23) a) -> void;

// --- fail_todo_import_unsupported_primitive_type.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_primitive_type.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: _BitInt(23)` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(11);
  // CHECK:STDERR:   ^~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(11);
  //@dump-sem-ir-end
}

// ============================================================================
// Unsupported primitive type among params
// ============================================================================

// --- unsupported_primitive_type_among_params.h

auto foo(int a, _BitInt(23) b) -> void;

// --- fail_todo_import_unsupported_primitive_type_among_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_primitive_type_among_params.h";

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type_among_params.carbon:[[@LINE+4]]:3: error: semantics TODO: `Unsupported: parameter type: _BitInt(23)` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1, 20);
  // CHECK:STDERR:   ^~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1, 20);
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_todo_import_unsupported_primitive_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_11: Core.IntLiteral = int_value 11 [concrete = constants.%int_11]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_todo_import_unsupported_primitive_type_among_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %.c5d: type = cpp_overload_set_type @<null name> [concrete]
// CHECK:STDOUT:   %empty_struct: %.c5d = struct_value () [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = %.a21
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %.a21: %.c5d = cpp_overload_set_value @<null name> [concrete = constants.%empty_struct]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: %.c5d = name_ref foo, imports.%.a21 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %int_20: Core.IntLiteral = int_value 20 [concrete = constants.%int_20]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
