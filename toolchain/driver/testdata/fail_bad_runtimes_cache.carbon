// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind --runtimes-cache=/a/b/c/does/not/exist compile --output=%t %s
//
// Diagnostic contains non-portable filesystem error text.
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/fail_bad_runtimes_cache.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/fail_bad_runtimes_cache.carbon
// CHECK:STDERR: error: Calling `openat` on '/a/b/c/does/not/exist' relative to 'AT_FDCWD' during DirRef::OpenDir failed: {{.*}} [DriverRuntimesCacheInvalid]
// CHECK:STDERR:
