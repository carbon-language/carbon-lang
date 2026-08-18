// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/primitives.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/macros/invalid.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/macros/invalid.carbon

// --- enums.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
// CHECK:STDERR: enums.carbon:[[@LINE+4]]:1: error: unknown type name 'error' [CppInteropParseError]
// CHECK:STDERR:     9 | error has_error;
// CHECK:STDERR:       | ^
// CHECK:STDERR:
error has_error;
#define macro_with_error has_error.member
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: enums.carbon:[[@LINE+11]]:3: error: failed to parse macro Cpp.macro_with_error to a valid constant expression [InCppMacroEvaluation]
  // CHECK:STDERR:   Cpp.macro_with_error;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR: enums.carbon:[[@LINE+8]]:3: note: in `Cpp` name lookup for `macro_with_error` [InCppNameLookup]
  // CHECK:STDERR:   Cpp.macro_with_error;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  // CHECK:STDERR: enums.carbon:[[@LINE+4]]:3: error: member name `macro_with_error` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:   Cpp.macro_with_error;
  // CHECK:STDERR:   ^~~~~~~~~~~~~~~~~~~~
  // CHECK:STDERR:
  Cpp.macro_with_error;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- enums.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: imports {
// CHECK:STDOUT:   %Cpp: <namespace> = namespace file.%Cpp.import_cpp, [concrete] {
// CHECK:STDOUT:     .macro_with_error = <poisoned>
// CHECK:STDOUT:     import Cpp//...
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %Cpp.ref: <namespace> = name_ref Cpp, imports.%Cpp [concrete = imports.%Cpp]
// CHECK:STDOUT:   %macro_with_error.ref: <error> = name_ref macro_with_error, <error> [concrete = <error>]
// CHECK:STDOUT:   <elided>
// CHECK:STDOUT: }
// CHECK:STDOUT:
