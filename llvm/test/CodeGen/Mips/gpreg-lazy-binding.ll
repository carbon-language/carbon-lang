; RUN: llc -march=mipsel -disable-mips-delay-filler < %s | FileCheck %s 

@g = external global i32

; CHECK:     addu  $gp
; CHECK:     jalr  $25
; CHECK:     nop
; CHECK-NOT: addu  $gp
; CHECK:     jalr  $25

define void @f0() nounwind {
entry:
  tail call void @externalFunc() nounwind
  tail call fastcc void @internalFunc()
  ret void
}

declare void @externalFunc()

define internal fastcc void @internalFunc() nounwind noinline {
entry:
  %0 = load i32* @g, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @g, align 4
  ret void
}

