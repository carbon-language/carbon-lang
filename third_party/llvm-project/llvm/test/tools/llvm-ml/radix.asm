; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.code

t1:
mov eax, 100b
mov eax, 100y

; CHECK-LABEL: t1:
; CHECK-NEXT: mov eax, 4
; CHECK-NEXT: mov eax, 4

t2:
mov eax, 100o
mov eax, 100q

; CHECK-LABEL: t2:
; CHECK-NEXT: mov eax, 64
; CHECK-NEXT: mov eax, 64

t3:
mov eax, 100d
mov eax, 100t

; CHECK-LABEL: t3:
; CHECK-NEXT: mov eax, 100
; CHECK-NEXT: mov eax, 100

t4:
mov eax, 100h

; CHECK-LABEL: t4:
; CHECK-NEXT: mov eax, 256

t5:
mov eax, 100
.radix 2
mov eax, 100
.radix 16
mov eax, 100
.radix 10
mov eax, 100

; CHECK-LABEL: t5:
; CHECK: mov eax, 100
; CHECK: mov eax, 4
; CHECK: mov eax, 256
; CHECK: mov eax, 100

t6:
.radix 9
mov eax, 100
.radix 10

; CHECK-LABEL: t6:
; CHECK: mov eax, 81

t7:
.radix 12
mov eax, 100b
mov eax, 100y
.radix 10

; CHECK-LABEL: t7:
; CHECK: mov eax, 1739
; CHECK: mov eax, 4

t8:
.radix 16
mov eax, 100d
mov eax, 100t
.radix 10

; CHECK-LABEL: t8:
; CHECK: mov eax, 4109
; CHECK: mov eax, 100

t9:
.radix 12
mov eax, 102b
.radix 16
mov eax, 10fd
.radix 10

; CHECK-LABEL: t9:
; CHECK: mov eax, 1763
; CHECK: mov eax, 4349

t10:
.radix 16
mov eax, 1e1
.radix 10

; CHECK-LABEL: t10:
; CHECK: mov eax, 481

END
