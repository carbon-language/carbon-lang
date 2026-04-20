// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/primitive_types/fail_todo_arithmetic_types_unmapped.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/primitive_types/fail_todo_arithmetic_types_unmapped.carbon

// Ensure we get a diagnostic when using a type that doesn't yet have a mapping
// from C++ to Carbon.

// --- fail_todo_wide_char_t.carbon

import Cpp inline '''
auto GetWC() -> wchar_t;
''';

// CHECK:STDERR: fail_todo_wide_char_t.carbon:[[@LINE+4]]:32: error: semantics TODO: `Unsupported: return type: wchar_t` [SemanticsTodo]
// CHECK:STDERR: fn CallGetWC() -> u32 { return Cpp.GetWC(); }
// CHECK:STDERR:                                ^~~~~~~~~~~
// CHECK:STDERR:
fn CallGetWC() -> u32 { return Cpp.GetWC(); }
