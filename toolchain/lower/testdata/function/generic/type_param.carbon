// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/type_param.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test --test_arg=--dump_output --test_arg=--file_tests=toolchain/lower/testdata/function/generic/type_param.carbon

fn F(T:! type) {
  var p: T*;
  let n: T = *p;
}

// CHECK:STDOUT: ; ModuleID = 'type_param.carbon'
// CHECK:STDOUT: source_filename = "type_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: %type = type {}
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F(%type %T) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %p.var = alloca ptr, align 8
// CHECK:STDOUT:   %.loc13_15 = load ptr, ptr %p.var, align 8
// CHECK:STDOUT:   %.loc13_14.2 = load {}, ptr %.loc13_15, align 1
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
