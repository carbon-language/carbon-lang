// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile foo.carbon -- bar.cpp
//
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/fail_clang_arg_extra_input.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/fail_clang_arg_extra_input.carbon
// CHECK:STDERR: error: unable to handle compilation, expected exactly one compiler job in ' "{{.*}}/clang" "-cc1" {{.*}} "-x" "c++" "bar.cpp";  "{{.*}}/clang" "-cc1" {{.*}} "-x" "c++" "<carbon Cpp imports>"; ' [CppInteropDriverError]
// CHECK:STDERR:

// --- foo.carbon

// --- bar.cpp
