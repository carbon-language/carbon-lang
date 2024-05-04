// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

fn G(p: i32*);

fn F() {
  var s: {.a: i32, .b: i32} = {.a = 1, .b = 2};
  G(&s.b);
}

// CHECK:STDOUT: ; ModuleID = 'address_of_field.carbon'
// CHECK:STDOUT: source_filename = "address_of_field.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.3.loc10_47 = internal constant { i32, i32 } { i32 1, i32 2 }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @G(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: define void @F() {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %s.var = alloca { i32, i32 }, align 8
// CHECK:STDOUT:   %.loc10_46.2.a = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 0
// CHECK:STDOUT:   %.loc10_46.4.b = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %s.var, ptr align 4 @struct.3.loc10_47, i64 8, i1 false)
// CHECK:STDOUT:   %.loc11_7.b = getelementptr inbounds { i32, i32 }, ptr %s.var, i32 0, i32 1
// CHECK:STDOUT:   call void @G(ptr %.loc11_7.b)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
