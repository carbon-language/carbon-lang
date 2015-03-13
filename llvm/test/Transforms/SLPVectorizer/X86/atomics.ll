; RUN: opt < %s -basicaa -slp-vectorizer -S |FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"

@x = global [4 x i32] zeroinitializer, align 16
@a = global [4 x i32] zeroinitializer, align 16

; The SLPVectorizer should not vectorize atomic stores and it should not
; schedule regular stores around atomic stores.

; CHECK-LABEL: test
; CHECK: store i32
; CHECK: store atomic i32
; CHECK: store i32
; CHECK: store atomic i32
; CHECK: store i32
; CHECK: store atomic i32
; CHECK: store i32
; CHECK: store atomic i32
define void @test() {
entry:
  store i32 0, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i64 0, i64 0), align 16
  store atomic i32 0, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @x, i64 0, i64 0) release, align 16
  store i32 0, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i64 0, i64 1), align 4
  store atomic i32 1, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @x, i64 0, i64 1) release, align 4
  store i32 0, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i64 0, i64 2), align 8
  store atomic i32 2, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @x, i64 0, i64 2) release, align 8
  store i32 0, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @a, i64 0, i64 3), align 4
  store atomic i32 3, i32* getelementptr inbounds ([4 x i32], [4 x i32]* @x, i64 0, i64 3) release, align 4
  ret void
}

