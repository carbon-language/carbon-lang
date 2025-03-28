// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --no-debug-info --phase=lower --dump-llvm-ir --output=- --exclude-dump-file-prefix=%{core_package_dir} %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/debug/nodebug.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/debug/nodebug.carbon

fn F() {
}

// CHECK:STDOUT: ; ModuleID = 'nodebug.carbon'
// CHECK:STDOUT: source_filename = "nodebug.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CF.Main() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
