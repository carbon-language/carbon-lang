// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=speed %s -- -###
//
// SET-CAPTURE-CONSOLE-OUTPUT
// SET-CHECK-SUBSET
// NOAUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/fail_clang_forward_optimize_speed.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/fail_clang_forward_optimize_speed.carbon
// CHECK:STDERR:  "{{.*}}/toolchain/install/prefix_root/lib/carbon/../../lib/carbon/llvm/bin/clang" "-cc1" {{.*}} "-O3" {{.*}}

// --- fail_foo.carbon
