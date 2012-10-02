; RUN: llc < %s -mcpu=cortex-a9 -verify-coalescing | FileCheck %s
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
declare <2 x float> @llvm.arm.neon.vld1.v2f32(i8*, i32) nounwind readonly

; CHECK: f4
; This function inserts a lane into a fully defined vector.
; The destination lane isn't read, so the subregs can coalesce.
; CHECK-NOT: vmov
; CHECK-NOT: vorr
define void @f4(float* %p, float* %q) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld1 = tail call <2 x float> @llvm.arm.neon.vld1.v2f32(i8* %0, i32 4)
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %1 = load float* %q, align 4
  %arrayidx1 = getelementptr inbounds float* %q, i32 1
  %2 = load float* %arrayidx1, align 4
  %add = fadd float %1, %2
  %vecins = insertelement <2 x float> %vld1, float %add, i32 1
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %x.0 = phi <2 x float> [ %vecins, %if.then ], [ %vld1, %entry ]
  tail call void @llvm.arm.neon.vst1.v2f32(i8* %0, <2 x float> %x.0, i32 4)
  ret void
}

; CHECK: f5
; Coalesce vector lanes through phis.
; CHECK: vmov.f32 {{.*}}, #1.0
; CHECK-NOT: vmov
; CHECK-NOT: vorr
; CHECK: %if.end
; We may leave the last insertelement in the if.end block.
; It is inserting the %add value into a dead lane, but %add causes interference
; in the entry block, and we don't do dead lane checks across basic blocks.
define void @f5(float* %p, float* %q) nounwind ssp {
entry:
  %0 = bitcast float* %p to i8*
  %vld1 = tail call <4 x float> @llvm.arm.neon.vld1.v4f32(i8* %0, i32 4)
  %vecext = extractelement <4 x float> %vld1, i32 0
  %vecext1 = extractelement <4 x float> %vld1, i32 1
  %vecext2 = extractelement <4 x float> %vld1, i32 2
  %vecext3 = extractelement <4 x float> %vld1, i32 3
  %add = fadd float %vecext3, 1.000000e+00
  %tobool = icmp eq float* %q, null
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %arrayidx = getelementptr inbounds float* %q, i32 1
  %1 = load float* %arrayidx, align 4
  %add4 = fadd float %vecext, %1
  %2 = load float* %q, align 4
  %add6 = fadd float %vecext1, %2
  %arrayidx7 = getelementptr inbounds float* %q, i32 2
  %3 = load float* %arrayidx7, align 4
  %add8 = fadd float %vecext2, %3
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %a.0 = phi float [ %add4, %if.then ], [ %vecext, %entry ]
  %b.0 = phi float [ %add6, %if.then ], [ %vecext1, %entry ]
  %c.0 = phi float [ %add8, %if.then ], [ %vecext2, %entry ]
  %vecinit = insertelement <4 x float> undef, float %a.0, i32 0
  %vecinit9 = insertelement <4 x float> %vecinit, float %b.0, i32 1
  %vecinit10 = insertelement <4 x float> %vecinit9, float %c.0, i32 2
  %vecinit11 = insertelement <4 x float> %vecinit10, float %add, i32 3
  tail call void @llvm.arm.neon.vst1.v4f32(i8* %0, <4 x float> %vecinit11, i32 4)
  ret void
}

declare <4 x float> @llvm.arm.neon.vld1.v4f32(i8*, i32) nounwind readonly

declare void @llvm.arm.neon.vst1.v4f32(i8*, <4 x float>, i32) nounwind

; CHECK: pr13999
define void @pr13999() nounwind readonly {
entry:
 br i1 true, label %outer_loop, label %loop.end

outer_loop:
 %d = phi double [ 0.0, %entry ], [ %add, %after_inner_loop ]
 %0 = insertelement <2 x double> <double 0.0, double 0.0>, double %d, i32 0
 br i1 undef, label %after_inner_loop, label %inner_loop

inner_loop:
 br i1 true, label %after_inner_loop, label %inner_loop

after_inner_loop:
 %1 = phi <2 x double> [ %0, %outer_loop ], [ <double 0.0, double 0.0>,
%inner_loop ]
 %2 = extractelement <2 x double> %1, i32 1
 %add = fadd double 1.0, %2
 br i1 false, label %loop.end, label %outer_loop

loop.end:
 %d.end = phi double [ 0.0, %entry ], [ %add, %after_inner_loop ]
 ret void
}
