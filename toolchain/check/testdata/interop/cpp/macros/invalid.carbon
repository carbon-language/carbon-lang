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

// --- fail_macro_with_error.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
// CHECK:STDERR: fail_macro_with_error.carbon:[[@LINE+2]]:1-5: error: unknown type name 'error' [CppInteropParseError]
// CHECK:STDERR:
error has_error;
#define macro_with_error has_error.member
''';

fn F() {
  //@dump-sem-ir-begin
  // CHECK:STDERR: fail_macro_with_error.carbon:[[@LINE+5]]:3-22: error: failed to parse macro Cpp.macro_with_error to a valid constant expression [InCppMacroEvaluation]
  // CHECK:STDERR: fail_macro_with_error.carbon:[[@LINE+4]]:3-22: note: in `Cpp` name lookup for `macro_with_error` [InCppNameLookup]
  // CHECK:STDERR:
  // CHECK:STDERR: fail_macro_with_error.carbon:[[@LINE+2]]:3-22: error: member name `macro_with_error` not found in `Cpp` [MemberNameNotFoundInInstScope]
  // CHECK:STDERR:
  Cpp.macro_with_error;
  //@dump-sem-ir-end
}

// CHECK:STDOUT: --- fail_macro_with_error.carbon
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
