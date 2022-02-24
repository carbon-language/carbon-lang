; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data

t1 BYTE NOT 1
; CHECK-LABEL: t1:
; CHECK-NEXT: .byte -2
; CHECK-NOT: .byte

t2 BYTE 1 OR 2
; CHECK-LABEL: t2:
; CHECK-NEXT: .byte 3

t3 BYTE 6 AND 10
; CHECK-LABEL: t3:
; CHECK-NEXT: .byte 2

t4 BYTE 5 EQ 6
   BYTE 6 EQ 6
   BYTE 7 EQ 6
; CHECK-LABEL: t4:
; CHECK-NEXT: .byte 0
; CHECK: .byte -1
; CHECK: .byte 0
; CHECK-NOT: .byte

t5 BYTE 5 NE 6
   BYTE 6 NE 6
   BYTE 7 NE 6
; CHECK-LABEL: t5:
; CHECK-NEXT: .byte -1
; CHECK: .byte 0
; CHECK: .byte -1
; CHECK-NOT: .byte

t6 BYTE 5 LT 6
   BYTE 6 LT 6
   BYTE 7 LT 6
; CHECK-LABEL: t6:
; CHECK-NEXT: .byte -1
; CHECK: .byte 0
; CHECK: .byte 0
; CHECK-NOT: .byte

t7 BYTE 5 LE 6
   BYTE 6 LE 6
   BYTE 7 LE 6
; CHECK-LABEL: t7:
; CHECK-NEXT: .byte -1
; CHECK: .byte -1
; CHECK: .byte 0
; CHECK-NOT: .byte

t8 BYTE 5 GT 6
   BYTE 6 GT 6
   BYTE 7 GT 6
; CHECK-LABEL: t8:
; CHECK-NEXT: .byte 0
; CHECK: .byte 0
; CHECK: .byte -1
; CHECK-NOT: .byte

t9 BYTE 5 GE 6
   BYTE 6 GE 6
   BYTE 7 GE 6
; CHECK-LABEL: t9:
; CHECK-NEXT: .byte 0
; CHECK: .byte -1
; CHECK: .byte -1
; CHECK-NOT: .byte

.code

t10:
xor eax, Not 1
; CHECK-LABEL: t10:
; CHECK-NEXT: xor eax, -2

t11:
xor eax, 1 oR 2
; CHECK-LABEL: t11:
; CHECK-NEXT: xor eax, 3

t12:
xor eax, 6 ANd 10
; CHECK-LABEL: t12:
; CHECK-NEXT: xor eax, 2

t13:
xor eax, 5 Eq 6
xor eax, 6 eQ 6
xor eax, 7 eq 6
; CHECK-LABEL: t13:
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, 0

t14:
xor eax, 5 Ne 6
xor eax, 6 nE 6
xor eax, 7 ne 6
; CHECK-LABEL: t14:
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, -1

t15:
xor eax, 5 Lt 6
xor eax, 6 lT 6
xor eax, 7 lt 6
; CHECK-LABEL: t15:
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, 0

t16:
xor eax, 5 Le 6
xor eax, 6 lE 6
xor eax, 7 le 6
; CHECK-LABEL: t16:
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, 0

t17:
xor eax, 5 Gt 6
xor eax, 6 gT 6
xor eax, 7 gt 6
; CHECK-LABEL: t17:
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, -1

t18:
xor eax, 5 Ge 6
xor eax, 6 gE 6
xor eax, 7 ge 6
; CHECK-LABEL: t18:
; CHECK-NEXT: xor eax, 0
; CHECK-NEXT: xor eax, -1
; CHECK-NEXT: xor eax, -1

END
