; RUN: llc -march=mips < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips -regalloc=basic < %s | FileCheck %s -check-prefix=O32
; RUN: llc -march=mips64el -mcpu=mips64 -mattr=n64 < %s | FileCheck %s -check-prefix=N64

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global i32* null, align 4

; O32-DAG:  lw $[[R0:[0-9]+]], %got(i3)
; O32-DAG:  addiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got(i1)
; O32:      movn $[[R0]], $[[R1]], ${{[0-9]+}}
; N64-DAG:  ldr $[[R0:[0-9]+]]
; N64-DAG:  ld $[[R1:[0-9]+]], %got_disp(i1)
; N64:      movn $[[R0]], $[[R1]], ${{[0-9]+}}
define i32* @cmov1(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32** @i3, align 4
  %cond = select i1 %tobool, i32* getelementptr inbounds ([3 x i32]* @i1, i32 0, i32 0), i32* %tmp1
  ret i32* %cond
}

@c = global i32 1, align 4
@d = global i32 0, align 4

; O32-LABEL: cmov2:
; O32: addiu $[[R1:[0-9]+]], ${{[a-z0-9]+}}, %got(d)
; O32: addiu $[[R0:[0-9]+]], ${{[a-z0-9]+}}, %got(c)
; O32: movn  $[[R1]], $[[R0]], ${{[0-9]+}}
; N64-LABEL: cmov2:
; N64: daddiu $[[R1:[0-9]+]], ${{[0-9]+}}, %got_disp(d)
; N64: daddiu $[[R0:[0-9]+]], ${{[0-9]+}}, %got_disp(c)
; N64: movn  $[[R1]], $[[R0]], ${{[0-9]+}}
define i32 @cmov2(i32 %s) nounwind readonly {
entry:
  %tobool = icmp ne i32 %s, 0
  %tmp1 = load i32* @c, align 4
  %tmp2 = load i32* @d, align 4
  %cond = select i1 %tobool, i32 %tmp1, i32 %tmp2
  ret i32 %cond
}

; O32-LABEL: cmov3:
; O32: xori $[[R0:[0-9]+]], ${{[0-9]+}}, 234
; O32: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]
define i32 @cmov3(i32 %a, i32 %b, i32 %c) nounwind readnone {
entry:
  %cmp = icmp eq i32 %a, 234
  %cond = select i1 %cmp, i32 %b, i32 %c
  ret i32 %cond
}

; N64-LABEL: cmov4:
; N64: xori $[[R0:[0-9]+]], ${{[0-9]+}}, 234
; N64: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]
define i64 @cmov4(i32 %a, i64 %b, i64 %c) nounwind readnone {
entry:
  %cmp = icmp eq i32 %a, 234
  %cond = select i1 %cmp, i64 %b, i64 %c
  ret i64 %cond
}

; slti and conditional move.
;
; Check that, pattern
;  (select (setgt a, N), t, f)
; turns into
;  (movz t, (setlt a, N + 1), f)
; if N + 1 fits in 16-bit.

; O32-LABEL: slti0:
; O32: slti $[[R0:[0-9]+]], ${{[0-9]+}}, 32767
; O32: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i32 @slti0(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, 32766
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: slti1:
; O32: slt ${{[0-9]+}}

define i32 @slti1(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, 32767
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: slti2:
; O32: slti $[[R0:[0-9]+]], ${{[0-9]+}}, -32768
; O32: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i32 @slti2(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, -32769
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: slti3:
; O32: slt ${{[0-9]+}}

define i32 @slti3(i32 %a) {
entry:
  %cmp = icmp sgt i32 %a, -32770
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; 64-bit patterns.

; N64-LABEL: slti64_0:
; N64: slti $[[R0:[0-9]+]], ${{[0-9]+}}, 32767
; N64: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i64 @slti64_0(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, 32766
  %conv = select i1 %cmp, i64 3, i64 4
  ret i64 %conv
}

; N64-LABEL: slti64_1:
; N64: slt ${{[0-9]+}}

define i64 @slti64_1(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, 32767
  %conv = select i1 %cmp, i64 3, i64 4
  ret i64 %conv
}

; N64-LABEL: slti64_2:
; N64: slti $[[R0:[0-9]+]], ${{[0-9]+}}, -32768
; N64: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i64 @slti64_2(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, -32769
  %conv = select i1 %cmp, i64 3, i64 4
  ret i64 %conv
}

; N64-LABEL: slti64_3:
; N64: slt ${{[0-9]+}}

define i64 @slti64_3(i64 %a) {
entry:
  %cmp = icmp sgt i64 %a, -32770
  %conv = select i1 %cmp, i64 3, i64 4
  ret i64 %conv
}

; sltiu instructions.

; O32-LABEL: sltiu0:
; O32: sltiu $[[R0:[0-9]+]], ${{[0-9]+}}, 32767
; O32: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i32 @sltiu0(i32 %a) {
entry:
  %cmp = icmp ugt i32 %a, 32766
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: sltiu1:
; O32: sltu ${{[0-9]+}}

define i32 @sltiu1(i32 %a) {
entry:
  %cmp = icmp ugt i32 %a, 32767
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: sltiu2:
; O32: sltiu $[[R0:[0-9]+]], ${{[0-9]+}}, -32768
; O32: movz ${{[0-9]+}}, ${{[0-9]+}}, $[[R0]]

define i32 @sltiu2(i32 %a) {
entry:
  %cmp = icmp ugt i32 %a, -32769
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}

; O32-LABEL: sltiu3:
; O32: sltu ${{[0-9]+}}

define i32 @sltiu3(i32 %a) {
entry:
  %cmp = icmp ugt i32 %a, -32770
  %cond = select i1 %cmp, i32 3, i32 4
  ret i32 %cond
}
