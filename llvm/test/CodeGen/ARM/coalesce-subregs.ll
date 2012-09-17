; RUN: llc < %s -mcpu=cortex-a9 -new-coalescer | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"
target triple = "thumbv7-apple-ios0.0.0"

; CHECK: f
; The vld2 and vst2 are not aligned wrt each other, the second Q loaded is the
; first one stored.
; The coalescer must find a super-register larger than QQ to eliminate the copy
; setting up the vst2 data.
; CHECK: vld2
; CHECK-NOT: vorr
; CHECK-NOT: vmov
; CHECK: vst2
define void @f(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %0, i32 4)
  %vld221 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %add.ptr = getelementptr inbounds float* %p, i32 8
  %1 = bitcast float* %add.ptr to i8*
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %1, <4 x float> %vld221, <4 x float> undef, i32 4)
  ret void
}

; CHECK: f1
; FIXME: This function still has copies.
define void @f1(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %0, i32 4)
  %vld221 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  %add.ptr = getelementptr inbounds float* %p, i32 8
  %1 = bitcast float* %add.ptr to i8*
  %vld22 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %1, i32 4)
  %vld2215 = extractvalue { <4 x float>, <4 x float> } %vld22, 0
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %1, <4 x float> %vld221, <4 x float> %vld2215, i32 4)
  ret void
}

; CHECK: f2
; FIXME: This function still has copies.
define void @f2(float* %p, i32 %c) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld2 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %0, i32 4)
  %vld224 = extractvalue { <4 x float>, <4 x float> } %vld2, 1
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  %qq0.0.1.0 = phi <4 x float> [ %vld224, %entry ], [ %vld2216, %do.body ]
  %c.addr.0 = phi i32 [ %c, %entry ], [ %dec, %do.body ]
  %p.addr.0 = phi float* [ %p, %entry ], [ %add.ptr, %do.body ]
  %add.ptr = getelementptr inbounds float* %p.addr.0, i32 8
  %1 = bitcast float* %add.ptr to i8*
  %vld22 = tail call { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8* %1, i32 4)
  %vld2215 = extractvalue { <4 x float>, <4 x float> } %vld22, 0
  %vld2216 = extractvalue { <4 x float>, <4 x float> } %vld22, 1
  tail call void @llvm.arm.neon.vst2.v4f32(i8* %1, <4 x float> %qq0.0.1.0, <4 x float> %vld2215, i32 4)
  %dec = add nsw i32 %c.addr.0, -1
  %tobool = icmp eq i32 %dec, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  ret void
}

declare { <4 x float>, <4 x float> } @llvm.arm.neon.vld2.v4f32(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst2.v4f32(i8*, <4 x float>, <4 x float>, i32) nounwind

; CHECK: f3
; This function has lane insertions that span basic blocks.
; The trivial REG_SEQUENCE lowering can't handle that, but the coalescer can.
;
; void f3(float *p, float *q) {
;   float32x2_t x;
;   x[1] = p[3];
;   if (q)
;     x[0] = q[0] + q[1];
;   else
;     x[0] = p[2];
;   vst1_f32(p+4, x);
; }
;
; CHECK-NOT: vmov
; CHECK-NOT: vorr
define void @f3(float* %p, float* %q) nounwind ssp {
entry:
  %arrayidx = getelementptr inbounds float* %p, i32 3
  %0 = load float* %arrayidx, align 4
  %vecins = insertelement <2 x float> undef, float %0, i32 1
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %1 = load float* %q, align 4
  %arrayidx2 = getelementptr inbounds float* %q, i32 1
  %2 = load float* %arrayidx2, align 4
  %add = fadd float %1, %2
  %vecins3 = insertelement <2 x float> %vecins, float %add, i32 0
  br label %if.end

if.else:                                          ; preds = %entry
  %arrayidx4 = getelementptr inbounds float* %p, i32 2
  %3 = load float* %arrayidx4, align 4
  %vecins5 = insertelement <2 x float> %vecins, float %3, i32 0
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %x.0 = phi <2 x float> [ %vecins3, %if.then ], [ %vecins5, %if.else ]
  %add.ptr = getelementptr inbounds float* %p, i32 4
  %4 = bitcast float* %add.ptr to i8*
  tail call void @llvm.arm.neon.vst1.v2f32(i8* %4, <2 x float> %x.0, i32 4)
  ret void
}

declare void @llvm.arm.neon.vst1.v2f32(i8*, <2 x float>, i32) nounwind
