// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/format/testdata/wrapping/call_args.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/format/testdata/wrapping/call_args.carbon

// --- test.carbon

fn F() {
  RegisterHandler(event_name, default_priority, my_callback, extra_argument_value);
}

// --- AUTOUPDATE-SPLIT

// CHECK:STDOUT: fn F() {
// CHECK:STDOUT:   RegisterHandler(event_name, default_priority, my_callback,
// CHECK:STDOUT:                   extra_argument_value);
// CHECK:STDOUT: }
