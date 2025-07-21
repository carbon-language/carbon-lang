// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/uint.carbon
// EXTRA-ARGS: --target=x86_64-linux-gnu
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/fail_arithmetic_types_unmapped.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/fail_arithmetic_types_unmapped.carbon

// --- u64_types.h

using UL = unsigned long;
UL GetUL();

using ULL = unsigned long long;
ULL GetULL();

// --- fail_todo_use_u64_types.carbon

import Cpp library "u64_types.h";

fn CallGetUL() -> u64 { return Cpp.GetUL(); }

// TODO: Eventually, `unsigned long long` should map to
// `Cpp.unsigned_long_long`, which should implicitly convert to `u64`.
// CHECK:STDERR: fail_todo_use_u64_types.carbon:[[@LINE+7]]:33: error: semantics TODO: `Unsupported: return type: ULL` [SemanticsTodo]
// CHECK:STDERR: fn CallGetULL() -> u64 { return Cpp.GetULL(); }
// CHECK:STDERR:                                 ^~~~~~~~~~
// CHECK:STDERR: fail_todo_use_u64_types.carbon:[[@LINE+4]]:33: note: in `Cpp` name lookup for `GetULL` [InCppNameLookup]
// CHECK:STDERR: fn CallGetULL() -> u64 { return Cpp.GetULL(); }
// CHECK:STDERR:                                 ^~~~~~~~~~
// CHECK:STDERR:
fn CallGetULL() -> u64 { return Cpp.GetULL(); }
