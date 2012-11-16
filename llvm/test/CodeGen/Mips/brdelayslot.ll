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

; Check that cvt.d.w goes into jalr's delay slot.
;
define void @foo3(i32 %a) nounwind {
entry:
; Default:     foo3:
; Default:     jalr
; Default:     cvt.d.w

  %conv = sitofp i32 %a to double
  tail call void @foo4(double %conv) nounwind
  ret void
}

declare void @foo4(double)

@g2 = external global i32
@g1 = external global i32
@g3 = external global i32

; Check that branch delay slot can be filled with an instruction with operand
; $1.
;
; Default:     foo5:
; Default-NOT: nop

define void @foo5(i32 %a) nounwind {
entry:
  %0 = load i32* @g2, align 4
  %tobool = icmp eq i32 %a, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:
  %1 = load i32* @g1, align 4
  %add = add nsw i32 %1, %0
  store i32 %add, i32* @g1, align 4
  br label %if.end

if.else:
  %2 = load i32* @g3, align 4
  %sub = sub nsw i32 %2, %0
  store i32 %sub, i32* @g3, align 4
  br label %if.end

if.end:
  ret void
}

