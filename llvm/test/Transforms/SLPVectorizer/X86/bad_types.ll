; RUN: opt < %s -basicaa -slp-vectorizer -S -mcpu=corei7-avx | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @test1(x86_mmx %a, x86_mmx %b, i64* %ptr) {
; Ensure we can handle x86_mmx values which are primitive and can be bitcast
; with integer types but can't be put into a vector.
;
; CHECK-LABEL: @test1
; CHECK:         store i64
; CHECK:         store i64
; CHECK:         ret void
entry:
  %a.cast = bitcast x86_mmx %a to i64
  %b.cast = bitcast x86_mmx %b to i64
  %a.and = and i64 %a.cast, 42
  %b.and = and i64 %b.cast, 42
  %gep = getelementptr i64, i64* %ptr, i32 1
  store i64 %a.and, i64* %ptr
  store i64 %b.and, i64* %gep
  ret void
}

define void @test2(x86_mmx %a, x86_mmx %b) {
; Same as @test1 but using phi-input vectorization instead of store
; vectorization.
;
; CHECK-LABEL: @test2
; CHECK:         and i64
; CHECK:         and i64
; CHECK:         ret void
entry:
  br i1 undef, label %if.then, label %exit

if.then:
  %a.cast = bitcast x86_mmx %a to i64
  %b.cast = bitcast x86_mmx %b to i64
  %a.and = and i64 %a.cast, 42
  %b.and = and i64 %b.cast, 42
  br label %exit

exit:
  %a.phi = phi i64 [ 0, %entry ], [ %a.and, %if.then ]
  %b.phi = phi i64 [ 0, %entry ], [ %b.and, %if.then ]
  tail call void @f(i64 %a.phi, i64 %b.phi)
  ret void
}

define i8 @test3(i8 *%addr) {
; Check that we do not vectorize types that are padded to a bigger ones.
;
; CHECK-LABEL: @test3
; CHECK-NOT:   <4 x i2>
; CHECK:       ret i8
entry:
  %a = bitcast i8* %addr to i2*
  %a0 = getelementptr inbounds i2, i2* %a, i64 0
  %a1 = getelementptr inbounds i2, i2* %a, i64 1
  %a2 = getelementptr inbounds i2, i2* %a, i64 2
  %a3 = getelementptr inbounds i2, i2* %a, i64 3
  %l0 = load i2, i2* %a0, align 1
  %l1 = load i2, i2* %a1, align 1
  %l2 = load i2, i2* %a2, align 1
  %l3 = load i2, i2* %a3, align 1
  br label %bb1
bb1:                                              ; preds = %entry
  %p0 = phi i2 [ %l0, %entry ]
  %p1 = phi i2 [ %l1, %entry ]
  %p2 = phi i2 [ %l2, %entry ]
  %p3 = phi i2 [ %l3, %entry ]
  %r  = zext i2 %p2 to i8
  ret i8 %r
}

declare void @f(i64, i64)
