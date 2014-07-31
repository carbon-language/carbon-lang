; RUN: opt -S -loop-vectorize -force-vector-unroll=1 -force-vector-width=2 -pass-remarks-analysis=loop-vectorize < %s 2>&1 | FileCheck %s

; CHECK: remark: {{.*}}: loop not vectorized: value could not be identified as an induction or reduction variable
; CHECK: remark: {{.*}}: loop not vectorized: use of induction value outside of the loop is not handled by vectorizer

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32-S128"

@f = common global i32 0, align 4
@.str = private unnamed_addr constant [4 x i8] c"%d\0A\00", align 1
@c = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4
@e = common global i32 0, align 4

; We used to vectorize this loop. But it has a value that is used outside of the
; and is not a recognized reduction variable "tmp17".

; CHECK-LABEL: @main(
; CHECK-NOT: <2 x i32>

define i32 @main()  {
bb:
  %b.promoted = load i32* @b, align 4
  br label %.lr.ph.i

.lr.ph.i:
  %tmp8 = phi i32 [ %tmp18, %bb16 ], [ %b.promoted, %bb ]
  %tmp2 = icmp sgt i32 %tmp8, 10
  br i1 %tmp2, label %bb16, label %bb10

bb10:
  br label %bb16

bb16:
  %tmp17 = phi i32 [ 0, %bb10 ], [ 1, %.lr.ph.i ]
  %tmp18 = add nsw i32 %tmp8, 1
  %tmp19 = icmp slt i32 %tmp18, 4
  br i1 %tmp19, label %.lr.ph.i, label %f1.exit.loopexit

f1.exit.loopexit:
  %.lcssa = phi i32 [ %tmp17, %bb16 ]
  ret i32 %.lcssa
}

; Don't vectorize this loop. Its phi node (induction variable) has an outside
; loop user. We currently don't handle this case.
; PR17179

; CHECK-LABEL: @test2(
; CHECK-NOT:  <2 x

@x1 = common global i32 0, align 4
@x2 = common global i32 0, align 4
@x0 = common global i32 0, align 4

define i32 @test2()  {
entry:
  store i32 0, i32* @x1, align 4
  %0 = load i32* @x0, align 4
  br label %for.cond1.preheader

for.cond1.preheader:
  %inc7 = phi i32 [ 0, %entry ], [ %inc, %for.cond1.preheader ]
  %inc = add nsw i32 %inc7, 1
  %cmp = icmp eq i32 %inc, 52
  br i1 %cmp, label %for.end5, label %for.cond1.preheader

for.end5:
  %inc7.lcssa = phi i32 [ %inc7, %for.cond1.preheader ]
  %xor = xor i32 %inc7.lcssa, %0
  store i32 52, i32* @x1, align 4
  store i32 1, i32* @x2, align 4
  ret i32 %xor
}
