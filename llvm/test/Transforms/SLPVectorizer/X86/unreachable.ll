; RUN: opt < %s -basicaa -slp-vectorizer -S -mtriple=x86_64-apple-macosx10.8.0 -mcpu=corei7

; Check if the SLPVectorizer does not crash when handling
; unreachable blocks with unscheduleable instructions.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

define void @foo(i32* nocapture %x) #0 {
entry:
  br label %bb2

bb1:                                    ; an unreachable block
  %t3 = getelementptr inbounds i32* %x, i64 4
  %t4 = load i32* %t3, align 4
  %t5 = getelementptr inbounds i32* %x, i64 5
  %t6 = load i32* %t5, align 4
  %bad = fadd float %bad, 0.000000e+00  ; <- an instruction with self dependency,
                                        ;    but legal in unreachable code
  %t7 = getelementptr inbounds i32* %x, i64 6
  %t8 = load i32* %t7, align 4
  %t9 = getelementptr inbounds i32* %x, i64 7
  %t10 = load i32* %t9, align 4
  br label %bb2

bb2:
  %t1.0 = phi i32 [ %t4, %bb1 ], [ 2, %entry ]
  %t2.0 = phi i32 [ %t6, %bb1 ], [ 2, %entry ]
  %t3.0 = phi i32 [ %t8, %bb1 ], [ 2, %entry ]
  %t4.0 = phi i32 [ %t10, %bb1 ], [ 2, %entry ]
  store i32 %t1.0, i32* %x, align 4
  %t12 = getelementptr inbounds i32* %x, i64 1
  store i32 %t2.0, i32* %t12, align 4
  %t13 = getelementptr inbounds i32* %x, i64 2
  store i32 %t3.0, i32* %t13, align 4
  %t14 = getelementptr inbounds i32* %x, i64 3
  store i32 %t4.0, i32* %t14, align 4
  ret void
}

