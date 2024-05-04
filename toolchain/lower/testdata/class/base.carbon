// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE

base class Base {
  var b: i32;
}

class Derived {
  extend base: Base;

  var d: i32;
}

fn Make() -> Derived {
  return {.base = {.b = 4}, .d = 7};
}

fn Access(d: Derived) -> (i32, i32) {
  return (d.d, d.base.b);
}

fn Convert(p: Derived*) -> Base* {
  return p;
}

// CHECK:STDOUT: ; ModuleID = 'base.carbon'
// CHECK:STDOUT: source_filename = "base.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @struct.3.loc18_36 = internal constant { { i32 }, i32 } { { i32 } { i32 4 }, i32 7 }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Make(ptr sret({ { i32 }, i32 }) %return) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_35.2.base = getelementptr inbounds { { i32 }, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   %.loc18_26.2.b = getelementptr inbounds { i32 }, ptr %.loc18_35.2.base, i32 0, i32 0
// CHECK:STDOUT:   %.loc18_35.4.d = getelementptr inbounds { { i32 }, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @struct.3.loc18_36, i64 8, i1 false)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @Access(ptr sret({ i32, i32 }) %return, ptr %d) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc22_12.1.d = getelementptr inbounds { { i32 }, i32 }, ptr %d, i32 0, i32 1
// CHECK:STDOUT:   %.loc22_12.2 = load i32, ptr %.loc22_12.1.d, align 4
// CHECK:STDOUT:   %.loc22_17.1.base = getelementptr inbounds { { i32 }, i32 }, ptr %d, i32 0, i32 0
// CHECK:STDOUT:   %.loc22_22.1.b = getelementptr inbounds { i32 }, ptr %.loc22_17.1.base, i32 0, i32 0
// CHECK:STDOUT:   %.loc22_22.2 = load i32, ptr %.loc22_22.1.b, align 4
// CHECK:STDOUT:   %.loc22_24.2.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 0
// CHECK:STDOUT:   store i32 %.loc22_12.2, ptr %.loc22_24.2.tuple.elem, align 4
// CHECK:STDOUT:   %.loc22_24.4.tuple.elem = getelementptr inbounds { i32, i32 }, ptr %return, i32 0, i32 1
// CHECK:STDOUT:   store i32 %.loc22_22.2, ptr %.loc22_24.4.tuple.elem, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @Convert(ptr %p) {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_11.2.base = getelementptr inbounds { { i32 }, i32 }, ptr %p, i32 0, i32 0
// CHECK:STDOUT:   ret ptr %.loc26_11.2.base
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
