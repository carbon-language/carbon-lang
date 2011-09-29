; RUN: llc -march=mipsel  -enable-mips-delay-filler < %s | FileCheck %s

define void @foo1() nounwind {
entry:
; CHECK:      jalr 
; CHECK-NOT:  nop 
; CHECK:      jr 
; CHECK-NOT:  nop
; CHECK:      .end

  tail call void @foo2(i32 3) nounwind
  ret void
}

declare void @foo2(i32)
