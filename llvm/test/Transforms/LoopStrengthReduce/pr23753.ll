; RUN: opt < %s -loop-reduce -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = external global i32, align 4
@d = external global i8, align 1

; CHECK-LABEL: void @f
define void @f() {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* @a, i64 %indvars.iv
  %cmp = icmp ne i32* %arrayidx, bitcast (i8* @d to i32*)
  %indvars.iv.next = add i64 %indvars.iv, 1
  br i1 %cmp, label %for.body, label %for.end

; CHECK:       %[[phi:.*]] = phi i8* [ %[[gep:.*]], {{.*}} ], [ getelementptr (i8, i8* @d, i64 sub (i64 4, i64 ptrtoint (i32* @a to i64))), {{.*}} ]
; CHECK-NEXT:  %[[gep]]    = getelementptr i8, i8* %[[phi]], i64 -4
; CHECK-NEXT:  %[[cst:.*]] = bitcast i8* %[[gep]] to i32*
; CHECK-NEXT:  %[[cmp:.*]] = icmp ne i32* %[[cst]], null
; CHECK-NEXT:  br i1 %[[cmp]]

for.end:
  ret void
}
