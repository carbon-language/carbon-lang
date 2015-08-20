; RUN: opt -S -basicaa -dse < %s | FileCheck %s

target datalayout = "E-p:64:64:64-a0:0:8-f32:32:32-f64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-v64:64:64-v128:128:128"

declare void @llvm.lifetime.start(i64, i8* nocapture) nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) nounwind
declare void @llvm.memset.p0i8.i8(i8* nocapture, i8, i8, i32, i1) nounwind
declare void @callee(i8*)

define void @test1() {
; CHECK-LABEL: @test1(
  %A = alloca i8

  store i8 0, i8* %A  ;; Written to by memset
; CHECK-NOT: store
  call void @llvm.lifetime.end(i64 1, i8* %A)
; CHECK-NOT: lifetime.end

  call void @llvm.memset.p0i8.i8(i8* %A, i8 0, i8 -1, i32 0, i1 false)
; CHECK-NOT: memset

  ret void
; CHECK: ret void
}

define void @test2(i32* %P) {
; CHECK-LABEL: test2
  %Q = getelementptr i32, i32* %P, i32 1
  %R = bitcast i32* %Q to i8*
  call void @llvm.lifetime.start(i64 4, i8* %R)
; CHECK: lifetime.start
  store i32 0, i32* %Q  ;; This store is dead.
; CHECK-NOT: store
  call void @llvm.lifetime.end(i64 4, i8* %R)
; CHECK: lifetime.end
  ret void
}

define void @test3(i8*) {
; CHECK-LABEL: test3
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, i8* %a)
; CHECK-NOT: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* %a)
; CHECK-NOT: lifetime.end
  call void @llvm.lifetime.start(i64 1, i8* undef)
; CHECK-NOT: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* undef)
; CHECK-NOT: lifetime.end
  ret void
}

define void @test4(i8*) {
; CHECK-LABEL: test4
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, i8* %a)
; CHECK: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* %a)
; CHECK: lifetime.end
  call void @llvm.lifetime.start(i64 1, i8* %0)
; CHECK: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* %0)
; CHECK: lifetime.end
  call void @llvm.lifetime.start(i64 1, i8* %a)
; CHECK-NOT: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* %a)
; CHECK-NOT: lifetime.end
  ret void
}

define void @test5() {
; CHECK-LABEL: test5
  %a = alloca i8
  %b = alloca i8
  call void @llvm.lifetime.start(i64 1, i8* %a)
; CHECK: lifetime.start
  call void @llvm.lifetime.end(i64 1, i8* %a)
; CHECK: lifetime.end
  call void @llvm.lifetime.start(i64 1, i8* %b)
; CHECK: lifetime.start
  call void @callee(i8* %b)
; CHECK: call void @callee
  call void @llvm.lifetime.end(i64 1, i8* %b)
; CHECK-NOT: lifetime.end
  ret void
}
