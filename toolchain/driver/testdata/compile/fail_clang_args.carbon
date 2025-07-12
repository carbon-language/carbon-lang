// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --target=x86-pc-linux-gnu --clang-arg=-Wall --clang-arg=-Wextra foo.carbon -- -Wuninitialized -Wno-all -###
//
// SET-CAPTURE-CONSOLE-OUTPUT
// SET-CHECK-SUBSET
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/fail_clang_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/fail_clang_args.carbon
// CHECK:STDERR: clang version {{.*}}
// CHECK:STDERR: InstalledDir: {{.*}}/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/llvm/bin
// CHECK:STDERR:  "{{.*}}/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/llvm/bin/clang" "-cc1" {{.*}}"-triple" "x86-pc-linux-gnu" {{.*}}"-fsyntax-only" {{.*}} "-resource-dir" {{.*}} "-Wall" "-Wextra" "-Wuninitialized" "-Wno-all" {{.*}}

// --- foo.carbon
