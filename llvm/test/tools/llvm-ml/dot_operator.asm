; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.data

FOO STRUCT
  a BYTE ?
  b BYTE ?
  c BYTE ?
  d BYTE ?
FOO ENDS

BAR STRUCT
  e WORD ?
  f WORD ?
BAR ENDS

var FOO <>

.code

t1:
mov al, var.a
mov al, var. b
mov al, var .c
mov al, var . d

; CHECK-LABEL: t1:
; CHECK: mov al, byte ptr [rip + var]
; CHECK: mov al, byte ptr [rip + var+1]
; CHECK: mov al, byte ptr [rip + var+2]
; CHECK: mov al, byte ptr [rip + var+3]

t2:
mov eax, FOO.a
mov ax, FOO. b
mov al, FOO .c
mov eax, FOO . d

; CHECK-LABEL: t2:
; CHECK: mov eax, 0
; CHECK: mov ax, 1
; CHECK: mov al, 2
; CHECK: mov eax, 3

t3:
mov al, BYTE PTR var[FOO.c]

; CHECK-LABEL: t3:
; CHECK: mov al, byte ptr [rip + var+2]

t4:
mov ax, var.BAR.f
mov ax, var .BAR.f
mov ax, var. BAR.f
mov ax, var.BAR .f
mov ax, var.BAR. f
mov ax, var . BAR . f

; CHECK-LABEL: t4:
; CHECK: mov ax, word ptr [rip + var+2]
; CHECK: mov ax, word ptr [rip + var+2]
; CHECK: mov ax, word ptr [rip + var+2]
; CHECK: mov ax, word ptr [rip + var+2]
; CHECK: mov ax, word ptr [rip + var+2]
; CHECK: mov ax, word ptr [rip + var+2]

END
