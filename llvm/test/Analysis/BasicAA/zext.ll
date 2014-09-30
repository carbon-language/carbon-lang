; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: test_with_zext
; CHECK:  NoAlias: i8* %a, i8* %b

define void @test_with_zext() {
  %1 = tail call i8* @malloc(i64 120)
  %a = getelementptr inbounds i8* %1, i64 8
  %2 = getelementptr inbounds i8* %1, i64 16
  %3 = zext i32 3 to i64
  %b = getelementptr inbounds i8* %2, i64 %3
  ret void
}

; CHECK-LABEL: test_with_lshr
; CHECK:  NoAlias: i8* %a, i8* %b

define void @test_with_lshr(i64 %i) {
  %1 = tail call i8* @malloc(i64 120)
  %a = getelementptr inbounds i8* %1, i64 8
  %2 = getelementptr inbounds i8* %1, i64 16
  %3 = lshr i64 %i, 2
  %b = getelementptr inbounds i8* %2, i64 %3
  ret void
}

; CHECK-LABEL: test_with_a_loop
; CHECK:  NoAlias: i8* %a, i8* %b

define void @test_with_a_loop() {
  %1 = tail call i8* @malloc(i64 120)
  %a = getelementptr inbounds i8* %1, i64 8
  %2 = getelementptr inbounds i8* %1, i64 16
  br label %for.loop

for.loop:
  %i = phi i32 [ 0, %0 ], [ %i.next, %for.loop ]
  %3 = zext i32 %i to i64
  %b = getelementptr inbounds i8* %2, i64 %3
  %i.next = add nuw nsw i32 %i, 1
  %4 = icmp eq i32 %i.next, 10
  br i1 %4, label %for.loop.exit, label %for.loop

for.loop.exit:
  ret void
}

; CHECK-LABEL: test_sign_extension
; CHECK:  PartialAlias: i64* %b.i64, i8* %a

define void @test_sign_extension(i32 %p) {
  %1 = tail call i8* @malloc(i64 120)
  %p.64 = zext i32 %p to i64
  %a = getelementptr inbounds i8* %1, i64 %p.64
  %p.minus1 = add i32 %p, -1
  %p.minus1.64 = zext i32 %p.minus1 to i64
  %b.i8 = getelementptr inbounds i8* %1, i64 %p.minus1.64
  %b.i64 = bitcast i8* %b.i8 to i64*
  ret void
}

; Function Attrs: nounwind
declare noalias i8* @malloc(i64)
