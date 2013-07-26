; RUN: opt -mem2reg -S -o - < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n8:16:32:64"

declare void @llvm.lifetime.start(i64 %size, i8* nocapture %ptr)
declare void @llvm.lifetime.end(i64 %size, i8* nocapture %ptr)

define void @test1() {
; Ensure we can look through a bitcast to i8* and the addition of lifetime
; markers.
;
; CHECK-LABEL: @test1(
; CHECK-NOT: alloca
; CHECK: ret void

  %A = alloca i32
  %B = bitcast i32* %A to i8*
  call void @llvm.lifetime.start(i64 2, i8* %B)
  store i32 1, i32* %A
  call void @llvm.lifetime.end(i64 2, i8* %B)
  ret void
}

define void @test2() {
; Ensure we can look through a GEP to i8* and the addition of lifetime
; markers.
;
; CHECK-LABEL: @test2(
; CHECK-NOT: alloca
; CHECK: ret void

  %A = alloca {i8, i16}
  %B = getelementptr {i8, i16}* %A, i32 0, i32 0
  call void @llvm.lifetime.start(i64 2, i8* %B)
  store {i8, i16} zeroinitializer, {i8, i16}* %A
  call void @llvm.lifetime.end(i64 2, i8* %B)
  ret void
}

define i32 @test3(i32 %x) {
; CHECK-LABEL: @test3(
;
; Check that we recursively walk the uses of the alloca and thus can see
; through round trip bitcasts, dead bitcasts, GEPs, multiple GEPs, and lifetime
; markers.
entry:
  %a = alloca i32
; CHECK-NOT: alloca

  %b = bitcast i32* %a to i8*
  %b2 = getelementptr inbounds i8* %b, i32 0
  %b3 = getelementptr inbounds i8* %b2, i32 0
  call void @llvm.lifetime.start(i64 -1, i8* %b3)
; CHECK-NOT: call void @llvm.lifetime.start

  store i32 %x, i32* %a
; CHECK-NOT: store

  %dead = bitcast i32* %a to i4096*
  %dead1 = bitcast i4096* %dead to i42*
  %dead2 = getelementptr inbounds i32* %a, i32 %x
; CHECK-NOT: bitcast
; CHECK-NOT: getelementptr

  %ret = load i32* %a
; CHECK-NOT: load

  ret i32 %ret
; CHECK: ret i32 %x
}
