; RUN: opt -S -dse < %s | FileCheck %s

declare void @llvm.memcpy.i8(i8*, i8*, i8, i32)
declare void @llvm.memmove.i8(i8*, i8*, i8, i32)
declare void @llvm.memset.i8(i8*, i8, i8, i32)

define void @test1() {
; CHECK: @test1
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memcpy
; CHECK-NOT: store

  call void @llvm.memcpy.i8(i8* %A, i8* %B, i8 -1, i32 0)

  ret void
; CHECK: ret void
}

define void @test2() {
; CHECK: @test2
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memmove
; CHECK-NOT: store

  call void @llvm.memmove.i8(i8* %A, i8* %B, i8 -1, i32 0)

  ret void
; CHECK: ret void
}

define void @test3() {
; CHECK: @test3
  %A = alloca i8
  %B = alloca i8

  store i8 0, i8* %A  ;; Written to by memset
; CHECK-NOT: store

  call void @llvm.memset.i8(i8* %A, i8 0, i8 -1, i32 0)

  ret void
; CHECK: ret void
}
