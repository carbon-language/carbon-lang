// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*);

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  // TODO: The lowering here is incorrect: we're incorrectly loading `s.b`.
  G(&s.b);
}

// CHECK:STDOUT: ; ModuleID = 'address_of_field.carbon'
// CHECK:STDOUT: source_filename = "address_of_field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT:   %s = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %struct = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %a = getelementptr inbounds { i32, i32 }, ptr %struct, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %a, align 4
// CHECK:STDOUT:   %b = getelementptr inbounds { i32, i32 }, ptr %struct, i32 0, i32 1
// CHECK:STDOUT:   store i32 2, ptr %b, align 4
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s, ptr align 4 %struct, i64 8, i1 false)
// CHECK:STDOUT:   %b1 = getelementptr inbounds { i32, i32 }, ptr %s, i32 0, i32 1
// CHECK:STDOUT:   %1 = load ptr, ptr %b1, align 8
// CHECK:STDOUT:   call void @G(ptr %1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
