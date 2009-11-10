; RUN: opt -S -dse < %s | FileCheck %s

declare void @llvm.lifetime.end(i64, i8*)
declare void @llvm.memset.i8(i8*, i8, i8, i32)

define void @test1() {
; CHECK: @test1
  %A = alloca i8

  store i8 0, i8* %A  ;; Written to by memset
  call void @llvm.lifetime.end(i64 1, i8* %A)
; CHECK: lifetime.end

  call void @llvm.memset.i8(i8* %A, i8 0, i8 -1, i32 0)
; CHECK-NOT: memset

  ret void
; CHECK: ret void
}
