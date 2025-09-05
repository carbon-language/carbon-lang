// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/fail_extern_c.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/fail_extern_c.carbon

// These tests were factored out of `extern_c.carbon` because we do not generate
// LLVM IR if any test contains errors. They should be moved back once they can
// successfully compile.

// ============================================================================
// extern "C" variable
// ============================================================================

// --- extern_c_variable.h

extern "C" int foo;

// --- fail_todo_import_extern_c_variable.carbon

library "[[@TEST_NAME]]";

// CHECK:STDERR: fail_todo_import_extern_c_variable.carbon:[[@LINE+4]]:10: in file included here [InCppInclude]
// CHECK:STDERR: ./extern_c_variable.h:2:16: error: semantics TODO: `Unsupported: Declaration type Var` [SemanticsTodo]
// CHECK:STDERR: extern "C" int foo;
// CHECK:STDERR:                ^
import Cpp library "extern_c_variable.h";

fn MyF() -> i32 {
  // CHECK:STDERR: fail_todo_import_extern_c_variable.carbon:[[@LINE+4]]:10: note: in `Cpp` name lookup for `foo` [InCppNameLookup]
  // CHECK:STDERR:   return Cpp.foo;
  // CHECK:STDERR:          ^~~~~~~
  // CHECK:STDERR:
  return Cpp.foo;
}

// ============================================================================
// extern "C" function with C++ special name
// ============================================================================

// --- extern_c_with_special_name.h

struct X {};

extern "C" X operator+(X, X);

// --- import_extern_c_with_special_name.carbon

library "[[@TEST_NAME]]";

import Cpp library "extern_c_with_special_name.h";

fn MyF(a: Cpp.X, b: Cpp.X) -> Cpp.X {
  return a + b;
}
