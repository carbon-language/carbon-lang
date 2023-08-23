// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn Run() -> i32 {
  var x: {.a: f64, .b: i32} = {.a = 0.0, .b = 1};
  var y: i32 = x.b;
  var z: i32 = y;
  return 0;
}

// CHECK:STDOUT: ; ModuleID = 'member_access.carbon'
// CHECK:STDOUT: source_filename = "member_access.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @Run() {
// CHECK:STDOUT:   %x = alloca { double, i32 }, align 8
// CHECK:STDOUT:   %struct = alloca { double, i32 }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { double, i32 }, ptr %struct, i32 0, i32 0
// CHECK:STDOUT:   store double 0.000000e+00, ptr %a, align 8
// CHECK:STDOUT:   %b = getelementptr inbounds { double, i32 }, ptr %struct, i32 0, i32 1
// CHECK:STDOUT:   store i32 1, ptr %b, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %x, ptr align 8 %struct, i64 16, i1 false)
// CHECK:STDOUT:   %y = alloca i32, align 4
// CHECK:STDOUT:   %b1 = getelementptr inbounds { double, i32 }, ptr %x, i32 0, i32 1
// CHECK:STDOUT:   %1 = load i32, ptr %b1, align 4
// CHECK:STDOUT:   store i32 %1, ptr %y, align 4
// CHECK:STDOUT:   %z = alloca i32, align 4
// CHECK:STDOUT:   %2 = load i32, ptr %y, align 4
// CHECK:STDOUT:   store i32 %2, ptr %z, align 4
// CHECK:STDOUT:   ret i32 0
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
