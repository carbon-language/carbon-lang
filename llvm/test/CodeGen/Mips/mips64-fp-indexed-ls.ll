; RUN: llc -march=mips64el -mcpu=mips64r2 -mattr=n64 < %s | FileCheck %s

%struct.S = type <{ [4 x float] }>

@s = external global [4 x %struct.S]
@gf = external global float
@gd = external global double

define float @foo0(float* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: lwxc1
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds float* %b, i64 %idxprom
  %0 = load float* %arrayidx, align 4
  ret float %0
}

define double @foo1(double* nocapture %b, i32 %o) nounwind readonly {
entry:
; CHECK: ldxc1
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds double* %b, i64 %idxprom
  %0 = load double* %arrayidx, align 8
  ret double %0
}

define float @foo2(i32 %b, i32 %c) nounwind readonly {
entry:
; CHECK: luxc1
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S]* @s, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  %0 = load float* %arrayidx2, align 1
  ret float %0
}

define void @foo3(float* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: swxc1
  %0 = load float* @gf, align 4
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds float* %b, i64 %idxprom
  store float %0, float* %arrayidx, align 4
  ret void
}

define void @foo4(double* nocapture %b, i32 %o) nounwind {
entry:
; CHECK: sdxc1
  %0 = load double* @gd, align 8
  %idxprom = zext i32 %o to i64
  %arrayidx = getelementptr inbounds double* %b, i64 %idxprom
  store double %0, double* %arrayidx, align 8
  ret void
}

define void @foo5(i32 %b, i32 %c) nounwind {
entry:
; CHECK: suxc1
  %0 = load float* @gf, align 4
  %idxprom = zext i32 %c to i64
  %idxprom1 = zext i32 %b to i64
  %arrayidx2 = getelementptr inbounds [4 x %struct.S]* @s, i64 0, i64 %idxprom1, i32 0, i64 %idxprom
  store float %0, float* %arrayidx2, align 1
  ret void
}

