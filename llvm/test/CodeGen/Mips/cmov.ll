; RUN: llc -march=mips -mcpu=4ke < %s | FileCheck %s

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global i32* null, align 4

; CHECK:  lw  $3, %got(i3)($gp)
; CHECK:  addiu $5, $gp, %got(i1)
define i32* @cmov1(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32** @i3, align 4
  %cond = select i1 %tobool, i32* getelementptr inbounds ([3 x i32]* @i1, i32 0, i32 0), i32* %tmp1
  ret i32* %cond
}

