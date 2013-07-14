; Test memset in cases where the set value is variable.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memset.p0i8.i32(i8 *nocapture, i8, i32, i32, i1) nounwind
declare void @llvm.memset.p0i8.i64(i8 *nocapture, i8, i64, i32, i1) nounwind

; No bytes, i32 version.
define void @f1(i8 *%dest, i8 %val) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 0, i32 1, i1 false)
  ret void
}

; No bytes, i64 version.
define void @f2(i8 *%dest, i8 %val) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 0, i32 1, i1 false)
  ret void
}

; 1 byte, i32 version.
define void @f3(i8 *%dest, i8 %val) {
; CHECK-LABEL: f3:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 1, i32 1, i1 false)
  ret void
}

; 1 byte, i64 version.
define void @f4(i8 *%dest, i8 %val) {
; CHECK-LABEL: f4:
; CHECK: stc %r3, 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 1, i32 1, i1 false)
  ret void
}

; 2 bytes, i32 version.
define void @f5(i8 *%dest, i8 %val) {
; CHECK-LABEL: f5:
; CHECK-DAG: stc %r3, 0(%r2)
; CHECK-DAG: stc %r3, 1(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 2, i32 1, i1 false)
  ret void
}

; 2 bytes, i64 version.
define void @f6(i8 *%dest, i8 %val) {
; CHECK-LABEL: f6:
; CHECK-DAG: stc %r3, 0(%r2)
; CHECK-DAG: stc %r3, 1(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 2, i32 1, i1 false)
  ret void
}

; 3 bytes, i32 version.
define void @f7(i8 *%dest, i8 %val) {
; CHECK-LABEL: f7:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(2,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 3, i32 1, i1 false)
  ret void
}

; 3 bytes, i64 version.
define void @f8(i8 *%dest, i8 %val) {
; CHECK-LABEL: f8:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(2,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 3, i32 1, i1 false)
  ret void
}

; 257 bytes, i32 version.
define void @f9(i8 *%dest, i8 %val) {
; CHECK-LABEL: f9:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 257, i32 1, i1 false)
  ret void
}

; 257 bytes, i64 version.
define void @f10(i8 *%dest, i8 %val) {
; CHECK-LABEL: f10:
; CHECK: stc %r3, 0(%r2)
; CHECK: mvc 1(256,%r2), 0(%r2)
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 257, i32 1, i1 false)
  ret void
}

; 258 bytes, i32 version.  258 bytes is too big for a single MVC.
; For now expect none, so that the test fails and gets updated when
; large copies are implemented.
define void @f11(i8 *%dest, i8 %val) {
; CHECK-LABEL: f11:
; CHECK-NOT: mvc
; CHECK: br %r14
  call void @llvm.memset.p0i8.i32(i8 *%dest, i8 %val, i32 258, i32 1, i1 false)
  ret void
}

; 258 bytes, i64 version, with the same comments as above.
define void @f12(i8 *%dest, i8 %val) {
; CHECK-LABEL: f12:
; CHECK-NOT: mvc
; CHECK: br %r14
  call void @llvm.memset.p0i8.i64(i8 *%dest, i8 %val, i64 258, i32 1, i1 false)
  ret void
}
