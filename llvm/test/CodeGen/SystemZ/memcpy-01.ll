; Test memcpy using MVC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8 *nocapture, i8 *nocapture, i32, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8 *nocapture, i8 *nocapture, i64, i32, i1) nounwind

define void @f1(i8 *%dest, i8 *%src) {
; CHECK: f1:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8 *%dest, i8 *%src, i32 0, i32 1,
                                       i1 false)
  ret void
}

define void @f2(i8 *%dest, i8 *%src) {
; CHECK: f2:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8 *%dest, i8 *%src, i64 0, i32 1,
                                       i1 false)
  ret void
}

define void @f3(i8 *%dest, i8 *%src) {
; CHECK: f3:
; CHECK: mvc 0(1,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8 *%dest, i8 *%src, i32 1, i32 1,
                                       i1 false)
  ret void
}

define void @f4(i8 *%dest, i8 *%src) {
; CHECK: f4:
; CHECK: mvc 0(1,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8 *%dest, i8 *%src, i64 1, i32 1,
                                       i1 false)
  ret void
}

define void @f5(i8 *%dest, i8 *%src) {
; CHECK: f5:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8 *%dest, i8 *%src, i32 256, i32 1,
                                       i1 false)
  ret void
}

define void @f6(i8 *%dest, i8 *%src) {
; CHECK: f6:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8 *%dest, i8 *%src, i64 256, i32 1,
                                       i1 false)
  ret void
}

; 257 bytes is too big for a single MVC.  For now expect none, so that
; the test fails and gets updated when large copies are implemented.
define void @f7(i8 *%dest, i8 *%src) {
; CHECK: f7:
; CHECK-NOT: mvc
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8 *%dest, i8 *%src, i32 257, i32 1,
                                       i1 false)
  ret void
}

define void @f8(i8 *%dest, i8 *%src) {
; CHECK: f8:
; CHECK-NOT: mvc
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8 *%dest, i8 *%src, i64 257, i32 1,
                                       i1 false)
  ret void
}
