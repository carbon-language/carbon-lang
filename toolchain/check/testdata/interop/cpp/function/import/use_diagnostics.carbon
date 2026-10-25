// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/function/import/use_diagnostics.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/function/import/use_diagnostics.carbon

// --- deprecated.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
[[deprecated("use Current")]] int Old() { return 1; }
template<typename T> [[deprecated("use Current")]] auto OldTemplate(T value) {
  return value;
}
''';

fn Call(value: i32) -> i32 {
  // CHECK:STDERR: deprecated.carbon:[[@LINE+3]]:25: warning: 'Old' is deprecated: use Current [CppInteropParseWarning]
  // CHECK:STDERR: deprecated.carbon:[[@LINE-8]]:3: note: 'Old' has been explicitly marked deprecated here [CppInteropParseNote]
  // CHECK:STDERR:
  let unused old: i32 = Cpp.Old();
  // CHECK:STDERR: deprecated.carbon:[[@LINE+3]]:10: warning: 'OldTemplate<int>' is deprecated: use Current [CppInteropParseWarning]
  // CHECK:STDERR: deprecated.carbon:[[@LINE-11]]:24: note: 'OldTemplate<int>' has been explicitly marked deprecated here [CppInteropParseNote]
  // CHECK:STDERR:
  return Cpp.OldTemplate(value);
}

// --- fail_unavailable.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''
__attribute__((unavailable("not supported"))) int Unavailable() { return 1; }
template<typename T>
__attribute__((unavailable("not supported"))) auto UnavailableTemplate(T value) {
  return value;
}
''';

fn Call(value: i32) {
  // CHECK:STDERR: fail_unavailable.carbon:[[@LINE+3]]:3: error: 'Unavailable' is unavailable: not supported [CppInteropParseError]
  // CHECK:STDERR: fail_unavailable.carbon:[[@LINE-9]]:51: note: 'Unavailable' has been explicitly marked unavailable here [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.Unavailable();
  // CHECK:STDERR: fail_unavailable.carbon:[[@LINE+3]]:3: error: 'UnavailableTemplate<int>' is unavailable: not supported [CppInteropParseError]
  // CHECK:STDERR: fail_unavailable.carbon:[[@LINE-11]]:52: note: 'UnavailableTemplate<int>' has been explicitly marked unavailable here [CppInteropParseNote]
  // CHECK:STDERR:
  Cpp.UnavailableTemplate(value);
}
