// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/fail_todo_arithmetic_types_unmapped.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/fail_todo_arithmetic_types_unmapped.carbon

// Ensure we get a diagnostic when using a type that doesn't yet have a mapping
// from C++ to Carbon.

// --- u64_types.h

unsigned long GetUL();

unsigned long long GetULL();

// --- fail_todo_use_u64_types.carbon

import Cpp library "u64_types.h";

fn CallGetUL() -> u64 { return Cpp.GetUL(); }

// TODO: Eventually, `unsigned long long` should map to
// `Cpp.unsigned_long_long`, which should implicitly convert to `u64`.
// We should switch to testing a different type when that case works.
// CHECK:STDERR: fail_todo_use_u64_types.carbon:[[@LINE+7]]:33: error: semantics TODO: `Unsupported: return type: unsigned long long` [SemanticsTodo]
// CHECK:STDERR: fn CallGetULL() -> u64 { return Cpp.GetULL(); }
// CHECK:STDERR:                                 ^~~~~~~~~~~~
// CHECK:STDERR: fail_todo_use_u64_types.carbon:[[@LINE+4]]:33: note: in call to Cpp function here [InCallToCppFunction]
// CHECK:STDERR: fn CallGetULL() -> u64 { return Cpp.GetULL(); }
// CHECK:STDERR:                                 ^~~~~~~~~~~~
// CHECK:STDERR:
fn CallGetULL() -> u64 { return Cpp.GetULL(); }
