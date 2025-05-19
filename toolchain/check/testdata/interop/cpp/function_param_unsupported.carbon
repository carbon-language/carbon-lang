// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1.1);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_todo_import_unsupported_primitive_type.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(1.1);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1.1);
}

// ============================================================================
// Unsupported primitive type among params
// ============================================================================

// --- unsupported_primitive_type_among_params.h

auto foo(int a, float b) -> void;

// --- fail_import_unsupported_primitive_type_among_params.carbon

library "[[@TEST_NAME]]";

import Cpp library "unsupported_primitive_type_among_params.h";

fn F() {
  // CHECK:STDERR: fail_import_unsupported_primitive_type_among_params.carbon:[[@LINE+7]]:3: error: semantics TODO: `Unsupported: parameter type: float` [SemanticsTodo]
  // CHECK:STDERR:   Cpp.foo(1, 2.0);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR: fail_import_unsupported_primitive_type_among_params.carbon:[[@LINE+4]]:3: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.foo(1, 2.0);
  // CHECK:STDERR:   ^~~~~~~
  // CHECK:STDERR:
  Cpp.foo(1, 2.0);
}

// CHECK:STDOUT: --- fail_todo_import_unsupported_primitive_type.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 1.1000000000000001 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "unsupported_primitive_type.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %float: f64 = float_literal 1.1000000000000001 [concrete = constants.%float]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: --- fail_import_unsupported_primitive_type_among_params.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type: type = fn_type @F [concrete]
// CHECK:STDOUT:   %F: %F.type = struct_value () [concrete]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete]
// CHECK:STDOUT:   %Int.type: type = generic_class_type @Int [concrete]
// CHECK:STDOUT:   %Int.generic: %Int.type = struct_value () [concrete]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(%int_32) [concrete]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete]
// CHECK:STDOUT:   %float: f64 = float_literal 2 [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Core: <namespace> = namespace file.%Core.import, [concrete] {
// CHECK:STDOUT:     .Int = %Core.Int
// CHECK:STDOUT:     import Core//prelude
// CHECK:STDOUT:     import Core//prelude/...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .foo = <error>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.Int: %Int.type = import_ref Core//prelude/types/int, Int, loaded [concrete = constants.%Int.generic]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .Core = imports.%Core
// CHECK:STDOUT:     .Cpp = imports.%Cpp
// CHECK:STDOUT:     .F = %F.decl
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %Core.import = import Core
// CHECK:STDOUT:   %Cpp.import_cpp = import_cpp {
// CHECK:STDOUT:     import Cpp "unsupported_primitive_type_among_params.h"
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl: %F.type = fn_decl @F [concrete = constants.%F] {} {}
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %int_32: Core.IntLiteral = int_value 32 [concrete = constants.%int_32]
// CHECK:STDOUT:   %i32: type = class_type @Int, @Int(constants.%int_32) [concrete = constants.%i32]
// CHECK:STDOUT:   %foo.ref: <error> = name_ref foo, <error> [concrete = <error>]
// CHECK:STDOUT:   %int_1: Core.IntLiteral = int_value 1 [concrete = constants.%int_1]
// CHECK:STDOUT:   %float: f64 = float_literal 2 [concrete = constants.%float]
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
