; RUN: opt -mtriple=arm-none-none-eabi -mcpu=cortex-m33 < %s -arm-parallel-dsp -S | FileCheck %s
;
; The loads are not consecutive: check that the rewrite isn't triggered.
;
; CHECK-NOT:  call i32 @llvm.arm.smlad
;
define dso_local i32 @test(i32 %arg, i32* nocapture readnone %arg1, i16* nocapture readonly %arg2, i16* nocapture readonly %arg3) {
entry:
  %cmp24 = icmp sgt i32 %arg, 0
  br i1 %cmp24, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %.pre = load i16, i16* %arg3, align 2
  %.pre27 = load i16, i16* %arg2, align 2
  br label %for.body

for.cond.cleanup:
  %mac1.0.lcssa = phi i32 [ 0, %entry ], [ %add11, %for.body ]
  ret i32 %mac1.0.lcssa

for.body:
  %mac1.026 = phi i32 [ %add11, %for.body ], [ 0, %for.body.preheader ]
  %i.025 = phi i32 [ %add, %for.body ], [ 0, %for.body.preheader ]
  %arrayidx = getelementptr inbounds i16, i16* %arg3, i32 %i.025
  %0 = load i16, i16* %arrayidx, align 2
  %add = add nuw nsw i32 %i.025, 1
  %arrayidx1 = getelementptr inbounds i16, i16* %arg3, i32 %add
  %1 = load i16, i16* %arrayidx1, align 2
  %arrayidx3 = getelementptr inbounds i16, i16* %arg2, i32 %i.025
  %2 = load i16, i16* %arrayidx3, align 2
  %conv = sext i16 %2 to i32
  %conv4 = sext i16 %0 to i32
  %mul = mul nsw i32 %conv, %conv4

; Here we add another constants offset of 2, to make sure the
; loads to %3 and %2 are not consecutive:

  %add5 = add nuw nsw i32 %i.025, 2
  %arrayidx6 = getelementptr inbounds i16, i16* %arg2, i32 %add5
  %3 = load i16, i16* %arrayidx6, align 2

  %conv7 = sext i16 %3 to i32
  %conv8 = sext i16 %1 to i32
  %mul9 = mul nsw i32 %conv7, %conv8
  %add10 = add i32 %mul, %mac1.026
  %add11 = add i32 %add10, %mul9
  %exitcond = icmp ne i32 %add, %arg
  br i1 %exitcond, label %for.body, label %for.cond.cleanup
}

