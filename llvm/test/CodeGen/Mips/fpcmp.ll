; RUN: llc  < %s -march=mipsel -mcpu=4ke | FileCheck %s -check-prefix=CHECK-MIPS32R2
; RUN: llc  < %s -march=mipsel | FileCheck %s -check-prefix=CHECK-MIPS1

@g1 = external global i32

define i32 @f(float %f0, float %f1) nounwind {
entry:
; CHECK-MIPS32R2: c.olt.s
; CHECK-MIPS32R2: movt
; CHECK-MIPS32R2: c.olt.s
; CHECK-MIPS32R2: movt
; CHECK-MIPS1: c.olt.s
; CHECK-MIPS1: bc1t
; CHECK-MIPS1: c.olt.s
; CHECK-MIPS1: bc1t
  %cmp = fcmp olt float %f0, %f1
  %conv = zext i1 %cmp to i32
  %tmp2 = load i32* @g1, align 4
  %add = add nsw i32 %tmp2, %conv
  store i32 %add, i32* @g1, align 4
  %cond = select i1 %cmp, i32 10, i32 20
  ret i32 %cond
}
