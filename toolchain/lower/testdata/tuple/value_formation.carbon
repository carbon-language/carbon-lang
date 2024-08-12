// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/tuple/value_formation.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/tuple/value_formation.carbon

fn G(ab: ((i32, i32, i32), (i32, i32, i32)));

fn F() {
  var a: (i32, i32, i32);
  var b: (i32, i32, i32);
  G((a, b));
}

// CHECK:STDOUT: ; ModuleID = 'value_formation.carbon'
// CHECK:STDOUT: source_filename = "value_formation.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %b.var = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %.loc16_6.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc16_6.2 = load i32, ptr %.loc16_6.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_6.3.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc16_6.4 = load i32, ptr %.loc16_6.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_6.5.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %a.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc16_6.6 = load i32, ptr %.loc16_6.5.tuple.elem, align 4
// CHECK:STDOUT:   %tuple.loc16_6 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc16_61 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc16_6.2, ptr %tuple.loc16_61, align 4
// CHECK:STDOUT:   %tuple.loc16_62 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc16_6.4, ptr %tuple.loc16_62, align 4
// CHECK:STDOUT:   %tuple.loc16_63 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_6, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc16_6.6, ptr %tuple.loc16_63, align 4
// CHECK:STDOUT:   %.loc16_9.1.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc16_9.2 = load i32, ptr %.loc16_9.1.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_9.3.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 1
// CHECK:STDOUT:   %.loc16_9.4 = load i32, ptr %.loc16_9.3.tuple.elem, align 4
// CHECK:STDOUT:   %.loc16_9.5.tuple.elem = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b.var, i32 0, i32 2
// CHECK:STDOUT:   %.loc16_9.6 = load i32, ptr %.loc16_9.5.tuple.elem, align 4
// CHECK:STDOUT:   %tuple.loc16_9 = alloca { i32, i32, i32 }, align 8
// CHECK:STDOUT:   %tuple.loc16_94 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc16_9.2, ptr %tuple.loc16_94, align 4
// CHECK:STDOUT:   %tuple.loc16_95 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc16_9.4, ptr %tuple.loc16_95, align 4
// CHECK:STDOUT:   %tuple.loc16_96 = getelementptr inbounds nuw { i32, i32, i32 }, ptr %tuple.loc16_9, i32 0, i32 2
// CHECK:STDOUT:   store i32 %.loc16_9.6, ptr %tuple.loc16_96, align 4
// CHECK:STDOUT:   %tuple.loc16_10 = alloca { ptr, ptr }, align 8
// CHECK:STDOUT:   %tuple.loc16_107 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_10, i32 0, i32 0
// CHECK:STDOUT:   store ptr %tuple.loc16_6, ptr %tuple.loc16_107, align 8
// CHECK:STDOUT:   %tuple.loc16_108 = getelementptr inbounds nuw { ptr, ptr }, ptr %tuple.loc16_10, i32 0, i32 1
// CHECK:STDOUT:   store ptr %tuple.loc16_9, ptr %tuple.loc16_108, align 8
// CHECK:STDOUT:   call void @G(ptr %tuple.loc16_10)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
