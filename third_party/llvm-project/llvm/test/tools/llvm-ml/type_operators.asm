; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

FOO STRUCT 2
  x BYTE ?
  y WORD 5 DUP (?)
FOO ENDS

.code

t1:
; CHECK-LABEL: t1:

mov eax, sizeof BYTE
mov eax, (sizeof sBYTE)
mov eax, sizeof(Db)
mov eax, type BYTE
mov eax, (type sBYTE)
mov eax, type(Db)
; CHECK: mov eax, 1
; CHECK: mov eax, 1
; CHECK: mov eax, 1
; CHECK: mov eax, 1
; CHECK: mov eax, 1
; CHECK: mov eax, 1

mov eax, sizeof(word)
mov eax, type(word)
; CHECK: mov eax, 2
; CHECK: mov eax, 2
mov eax, sizeof(dword)
mov eax, type(dword)
; CHECK: mov eax, 4
; CHECK: mov eax, 4
mov eax, sizeof(fword)
mov eax, type(fword)
; CHECK: mov eax, 6
; CHECK: mov eax, 6
mov eax, sizeof(qword)
mov eax, type(qword)
; CHECK: mov eax, 8
; CHECK: mov eax, 8

mov eax, sizeof(real4)
mov eax, type(real4)
; CHECK: mov eax, 4
; CHECK: mov eax, 4
mov eax, sizeof(real8)
mov eax, type(real8)
; CHECK: mov eax, 8
; CHECK: mov eax, 8

mov eax, sizeof(FOO)
mov eax, type(FOO)
; CHECK: mov eax, 12
; CHECK: mov eax, 12


t2_full BYTE "ab"
t2_short DB ?
t2_signed SBYTE 3 DUP (?)

t2:
; CHECK-LABEL: t2:

mov eax, sizeof(t2_full)
mov eax, lengthof(t2_full)
mov eax, type(t2_full)
; CHECK: mov eax, 2
; CHECK: mov eax, 2
; CHECK: mov eax, 1

mov eax, sizeof(t2_short)
mov eax, lengthof(t2_short)
mov eax, type(t2_short)
; CHECK: mov eax, 1
; CHECK: mov eax, 1
; CHECK: mov eax, 1

mov eax, sizeof(t2_signed)
mov eax, lengthof(t2_signed)
mov eax, type(t2_signed)
; CHECK: mov eax, 3
; CHECK: mov eax, 3
; CHECK: mov eax, 1


t3_full WORD 2 DUP (?)
t3_short DW ?
t3_signed SWORD 3 DUP (?)

t3:
; CHECK-LABEL: t3:

mov eax, sizeof(t3_full)
mov eax, lengthof(t3_full)
mov eax, type(t3_full)
; CHECK: mov eax, 4
; CHECK: mov eax, 2
; CHECK: mov eax, 2

mov eax, sizeof(t3_short)
mov eax, lengthof(t3_short)
mov eax, type(t3_short)
; CHECK: mov eax, 2
; CHECK: mov eax, 1
; CHECK: mov eax, 2

mov eax, sizeof(t3_signed)
mov eax, lengthof(t3_signed)
mov eax, type(t3_signed)
; CHECK: mov eax, 6
; CHECK: mov eax, 3
; CHECK: mov eax, 2


t4_full DWORD 2 DUP (?)
t4_short DD ?
t4_signed SDWORD 3 DUP (?)

t4:
; CHECK-LABEL: t4:

mov eax, sizeof(t4_full)
mov eax, lengthof(t4_full)
mov eax, type(t4_full)
; CHECK: mov eax, 8
; CHECK: mov eax, 2
; CHECK: mov eax, 4

mov eax, sizeof(t4_short)
mov eax, lengthof(t4_short)
mov eax, type(t4_short)
; CHECK: mov eax, 4
; CHECK: mov eax, 1
; CHECK: mov eax, 4

mov eax, sizeof(t4_signed)
mov eax, lengthof(t4_signed)
mov eax, type(t4_signed)
; CHECK: mov eax, 12
; CHECK: mov eax, 3
; CHECK: mov eax, 4


t5_full FWORD 2 DUP (?)
t5_short DF ?

t5:
; CHECK-LABEL: t5:

mov eax, sizeof(t5_full)
mov eax, lengthof(t5_full)
mov eax, type(t5_full)
; CHECK: mov eax, 12
; CHECK: mov eax, 2
; CHECK: mov eax, 6

mov eax, sizeof(t5_short)
mov eax, lengthof(t5_short)
mov eax, type(t5_short)
; CHECK: mov eax, 6
; CHECK: mov eax, 1
; CHECK: mov eax, 6


t6_full QWORD 2 DUP (?)
t6_short DQ ?
t6_signed SQWORD 3 DUP (?)

t6:
; CHECK-LABEL: t6:

mov eax, sizeof(t6_full)
mov eax, lengthof(t6_full)
mov eax, type(t6_full)
; CHECK: mov eax, 16
; CHECK: mov eax, 2
; CHECK: mov eax, 8

mov eax, sizeof(t6_short)
mov eax, lengthof(t6_short)
mov eax, type(t6_short)
; CHECK: mov eax, 8
; CHECK: mov eax, 1
; CHECK: mov eax, 8

mov eax, sizeof(t6_signed)
mov eax, lengthof(t6_signed)
mov eax, type(t6_signed)
; CHECK: mov eax, 24
; CHECK: mov eax, 3
; CHECK: mov eax, 8


t7_single REAL4 2 DUP (?)
t7_double REAL8 ?
t7_extended REAL10 3 DUP (?)

t7:
; CHECK-LABEL: t7:

mov eax, sizeof(t7_single)
mov eax, lengthof(t7_single)
mov eax, type(t7_single)
; CHECK: mov eax, 8
; CHECK: mov eax, 2
; CHECK: mov eax, 4

mov eax, sizeof(t7_double)
mov eax, lengthof(t7_double)
mov eax, type(t7_double)
; CHECK: mov eax, 8
; CHECK: mov eax, 1
; CHECK: mov eax, 8

mov eax, sizeof(t7_extended)
mov eax, lengthof(t7_extended)
mov eax, type(t7_extended)
; CHECK: mov eax, 30
; CHECK: mov eax, 3
; CHECK: mov eax, 10


t8_var FOO <>, <>

t8:
; CHECK-LABEL: t8:

mov eax, sizeof(t8_var)
mov eax, lengthof(t8_var)
mov eax, type(t8_var)
; CHECK: mov eax, 24
; CHECK: mov eax, 2
; CHECK: mov eax, 12

mov eax, sizeof(t8_var.y)
mov eax, lengthof(t8_var.y)
mov eax, type(t8_var.y)
; CHECK: mov eax, 10
; CHECK: mov eax, 5
; CHECK: mov eax, 2

END
