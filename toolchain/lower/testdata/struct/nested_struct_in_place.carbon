// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/struct/nested_struct_in_place.carbon

fn F() -> (i32, i32, i32);

fn G() {
  var v: {.a: (i32, i32, i32), .b: (i32, i32, i32)} = {.a = F(), .b = F()};
}

// CHECK:STDOUT: ; ModuleID = 'nested_struct_in_place.carbon'
// CHECK:STDOUT: source_filename = "nested_struct_in_place.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @F(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: define void @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %v.var = alloca { { i32, i32, i32 }, { i32, i32, i32 } }, align 8
// CHECK:STDOUT:   %.loc14_74.1.a = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 0
// CHECK:STDOUT:   call void @F(ptr %.loc14_74.1.a)
// CHECK:STDOUT:   %.loc14_74.2.b = getelementptr inbounds nuw { { i32, i32, i32 }, { i32, i32, i32 } }, ptr %v.var, i32 0, i32 1
// CHECK:STDOUT:   call void @F(ptr %.loc14_74.2.b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
