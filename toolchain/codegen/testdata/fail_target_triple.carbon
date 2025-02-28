// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-prelude-import --target=x86_687-unknown-linux-gnu --output=- %s
// No autoupdate because the message comes from LLVM.
// To test this file alone, run:
//   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/codegen/testdata/fail_target_triple.carbon
// To dump output, run:
//   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/codegen/testdata/fail_target_triple.carbon
// NOAUTOUPDATE
// CHECK:STDERR: error: invalid target: {{.*}}x86_687{{.*}}
// CHECK:STDERR:

fn Main() {}
