; RUN: llc -march=mipsel -O0 < %s | FileCheck %s -check-prefix=None
; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=Default

define void @foo1() nounwind {
entry:
; Default:     jalr 
; Default-NOT: nop 
; Default:     jr 
; Default-NOT: nop
; Default:     .end
; None: jalr 
; None: nop 
; None: jr 
; None: nop
; None: .end

  tail call void @foo2(i32 3) nounwind
  ret void
}

declare void @foo2(i32)
