; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -enable-if-conversion -dce -instcombine -licm -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; This is the loop in this example:
;
;int function0(int *a, int *b, int start, int end) {
;
;  for (int i=start; i<end; ++i) {
;    unsigned k = a[i];
;
;    if (a[i] > b[i])   <------ notice the IF inside the loop.
;      k = k * 5 + 3;
;
;    a[i] = k;  <---- K is a phi node that becomes vector-select.
;  }
;}

;CHECK: @function0
;CHECK: load <4 x i32>
;CHECK: icmp sgt <4 x i32>
;CHECK: mul <4 x i32>
;CHECK: add <4 x i32>
;CHECK: select <4 x i1>
;CHECK: ret i32
define i32 @function0(i32* nocapture %a, i32* nocapture %b, i32 %start, i32 %end) nounwind uwtable ssp {
entry:
  %cmp16 = icmp slt i32 %start, %end
  br i1 %cmp16, label %for.body.lr.ph, label %for.end

for.body.lr.ph:
  %0 = sext i32 %start to i64
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ %0, %for.body.lr.ph ], [ %indvars.iv.next, %if.end ]
  %arrayidx = getelementptr inbounds i32* %a, i64 %indvars.iv
  %1 = load i32* %arrayidx, align 4
  %arrayidx4 = getelementptr inbounds i32* %b, i64 %indvars.iv
  %2 = load i32* %arrayidx4, align 4
  %cmp5 = icmp sgt i32 %1, %2
  br i1 %cmp5, label %if.then, label %if.end

if.then:
  %mul = mul i32 %1, 5
  %add = add i32 %mul, 3
  br label %if.end

if.end:
  %k.0 = phi i32 [ %add, %if.then ], [ %1, %for.body ]
  store i32 %k.0, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %3 = trunc i64 %indvars.iv.next to i32
  %cmp = icmp slt i32 %3, %end
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret i32 undef
}
