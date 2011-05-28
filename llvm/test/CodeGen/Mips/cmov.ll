; RUN: llc -march=mips -mcpu=4ke < %s | FileCheck %s
; RUN: llc -march=mips -mcpu=4ke -regalloc=basic < %s | FileCheck %s

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global i32* null, align 4

; CHECK:  addiu ${{[0-9]+}}, $gp, %got(i1)
; CHECK:  lw  ${{[0-9]+}}, %got(i3)($gp)
define i32* @cmov1(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32** @i3, align 4
  %cond = select i1 %tobool, i32* getelementptr inbounds ([3 x i32]* @i1, i32 0, i32 0), i32* %tmp1
  ret i32* %cond
}

@c = global i32 1, align 4
@d = global i32 0, align 4

; CHECK: cmov2:
; CHECK: addiu $[[R0:[0-9]+]], $gp, %got(c)
; CHECK: addiu $[[R1:[0-9]+]], $gp, %got(d)
; CHECK: movn  $[[R1]], $[[R0]], ${{[0-9]+}}
define i32 @cmov2(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32* @c, align 4
  %tmp2 = load i32* @d, align 4
  %cond = select i1 %tobool, i32 %tmp1, i32 %tmp2
  ret i32 %cond
}

