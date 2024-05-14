// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn F(arr: [i32; 3], i: i32) -> i32 {
  return arr[i];
}

fn G() -> i32 {
  return F((1, 2, 3), 1);
}

// CHECK:STDOUT: ; ModuleID = 'function_param.carbon'
// CHECK:STDOUT: source_filename = "function_param.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @array.loc12_11.1 = internal constant [3 x i32] [i32 1, i32 2, i32 3]
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @F(ptr %arr, i32 %i) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_15.2.array.index = getelementptr inbounds [3 x i32], ptr %arr, i32 0, i32 %i
// CHECK:STDOUT:   %.loc8_15.3 = load i32, ptr %.loc8_15.2.array.index, align 4
// CHECK:STDOUT:   ret i32 %.loc8_15.3
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @G() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_20.2.temp = alloca [3 x i32], align 4
// CHECK:STDOUT:   %.loc12_20.4.array.index = getelementptr inbounds [3 x i32], ptr %.loc12_20.2.temp, i32 0, i32 0
// CHECK:STDOUT:   %.loc12_20.7.array.index = getelementptr inbounds [3 x i32], ptr %.loc12_20.2.temp, i32 0, i32 1
// CHECK:STDOUT:   %.loc12_20.10.array.index = getelementptr inbounds [3 x i32], ptr %.loc12_20.2.temp, i32 0, i32 2
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %.loc12_20.2.temp, ptr align 4 @array.loc12_11.1, i64 12, i1 false)
// CHECK:STDOUT:   %F.call = call i32 @F(ptr %.loc12_20.2.temp, i32 1)
// CHECK:STDOUT:   ret i32 %F.call
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
