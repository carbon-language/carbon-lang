; RUN: llc -march=mipsel -mcpu=mips32r2 < %s | FileCheck %s

%struct.S = type <{ [4 x float] }>

@s = external global [4 x %struct.S]
@gf = external global float
@gd = external global double

define float @foo0(float* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: lwxc1
  %arrayidx = getelementptr inbounds float* %b, i32 %o
  %0 = load float* %arrayidx, align 4
  ret float %0
}

define double @foo1(double* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: ldxc1
  %arrayidx = getelementptr inbounds double* %b, i32 %o
  %0 = load double* %arrayidx, align 8
  ret double %0
}

define float @foo2(i32 %b, i32 %c) nounwind readonly {
entry:
; CHECK: luxc1
  %arrayidx1 = getelementptr inbounds [4 x %struct.S]* @s, i32 0, i32 %b, i32 0, i32 %c
  %0 = load float* %arrayidx1, align 1
  ret float %0
}

define void @foo3(float* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: swxc1
  %0 = load float* @gf, align 4
  %arrayidx = getelementptr inbounds float* %b, i32 %o
  store float %0, float* %arrayidx, align 4
  ret void
}

define void @foo4(double* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: sdxc1
  %0 = load double* @gd, align 8
  %arrayidx = getelementptr inbounds double* %b, i32 %o
  store double %0, double* %arrayidx, align 8
  ret void
}

define void @foo5(i32 %b, i32 %c) nounwind {
entry:
; CHECK: suxc1
  %0 = load float* @gf, align 4
  %arrayidx1 = getelementptr inbounds [4 x %struct.S]* @s, i32 0, i32 %b, i32 0, i32 %c
  store float %0, float* %arrayidx1, align 1
  ret void
}

