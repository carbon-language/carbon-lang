; RUN: opt < %s  -loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -enable-if-conversion -dce -instcombine -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @foo(
;CHECK: icmp sgt
;CHECK: icmp sgt
;CHECK: icmp slt
;CHECK: select <4 x i1>
;CHECK: %[[P1:.*]] = select <4 x i1>
;CHECK: xor <4 x i1>
;CHECK: and <4 x i1>
;CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %[[P1]]
;CHECK: ret
define i32 @foo(i32* nocapture %A, i32* nocapture %B, i32 %n) {
entry:
  %cmp26 = icmp sgt i32 %n, 0
  br i1 %cmp26, label %for.body, label %for.end

for.body:
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end14 ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32* %A, i64 %indvars.iv
  %0 = load i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32* %B, i64 %indvars.iv
  %1 = load i32* %arrayidx2, align 4
  %cmp3 = icmp sgt i32 %0, %1
  br i1 %cmp3, label %if.then, label %if.end14

if.then:
  %cmp6 = icmp sgt i32 %0, 19
  br i1 %cmp6, label %if.end14, label %if.else

if.else:
  %cmp10 = icmp slt i32 %1, 4
  %. = select i1 %cmp10, i32 4, i32 5
  br label %if.end14

if.end14:
  %x.0 = phi i32 [ 9, %for.body ], [ 3, %if.then ], [ %., %if.else ]  ; <------------- A PHI with 3 entries that we can still vectorize.
  store i32 %x.0, i32* %arrayidx, align 4
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret i32 undef
}
