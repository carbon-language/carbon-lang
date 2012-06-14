; RUN: llc -march=mipsel -force-mips-long-branch < %s | FileCheck %s

@g0 = external global i32

define void @foo1(i32 %s) nounwind {
entry:
; CHECK: lw  $[[R0:[a-z0-9]+]], %got($BB0_3)(${{[a-z0-9]+}})
; CHECK: addiu $[[R1:[a-z0-9]+]], $[[R0]], %lo($BB0_3)
; CHECK: jr  $[[R1]]

  %tobool = icmp eq i32 %s, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  %0 = load i32* @g0, align 4
  %add = add nsw i32 %0, 12
  store i32 %add, i32* @g0, align 4
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

