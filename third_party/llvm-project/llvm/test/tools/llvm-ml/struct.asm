; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

.data
BAZ STRUCT
  a BYTE 1
  b BYTE 2
BAZ ENDS

FOOBAR struct 2
  c BYTE 3 DUP (4)
  d DWORD 5
  e BAZ <>
  STRUCT f
    g BYTE 6
    h BYTE 7
  ends
  h BYTE "abcde"
foobar ENDS

t1 foobar <>

; CHECK: t1:
;
; BYTE 3 DUP (4), plus alignment padding
; CHECK-NEXT: .byte 4
; CHECK-NEXT: .byte 4
; CHECK-NEXT: .byte 4
; CHECK-NEXT: .zero 1
;
; DWORD 5
; CHECK-NEXT: .long 5
;
; BAZ <>
; CHECK-NEXT: .byte 1
; CHECK-NEXT: .byte 2
;
; <BYTE 6, BYTE 7>, with no alignment padding (field size < alignment)
; CHECK-NEXT: .byte 6
; CHECK-NEXT: .byte 7
;
; BYTE "abcde", plus alignment padding
; CHECK-NEXT: .byte 97
; CHECK-NEXT: .byte 98
; CHECK-NEXT: .byte 99
; CHECK-NEXT: .byte 100
; CHECK-NEXT: .byte 101
; CHECK-NEXT: .zero 1

t2 FOOBAR <"gh",,<10,11>,<12>,'ijk'>

; CHECK: t2:
;
; BYTE "gh", padded with " ", plus alignment padding
; CHECK-NEXT: .byte 103
; CHECK-NEXT: .byte 104
; CHECK-NEXT: .byte 32
; CHECK-NEXT: .zero 1
;
; DWORD 5 (default-initialized when omitted)
; CHECK-NEXT: .long 5
;
; BAZ <10, 11>
; CHECK-NEXT: .byte 10
; CHECK-NEXT: .byte 11
;
; <BYTE 12, BYTE 7>, with no alignment padding (field size < alignment)
; CHECK-NEXT: .byte 12
; CHECK-NEXT: .byte 7
;
; BYTE "ijk", padded with " ", plus alignment padding
; CHECK-NEXT: .byte 105
; CHECK-NEXT: .byte 106
; CHECK-NEXT: .byte 107
; CHECK-NEXT: .byte 32
; CHECK-NEXT: .byte 32
; CHECK-NEXT: .zero 1

.code

t3:
mov al, t2.f.h
mov al, [t2].f.h
mov al, [t2.f.h]

; CHECK: t3:
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]

t4:
mov al, j.FOOBAR.f.h
mov al, j.baz.b

; CHECK: t4:
; CHECK-NEXT: mov al, byte ptr [rip + j+11]
; CHECK-NEXT: mov al, byte ptr [rip + j+1]

t5:
mov al, [ebx].FOOBAR.f.h
mov al, [ebx.FOOBAR].f.h
mov al, [ebx.FOOBAR.f.h]

; CHECK: t5:
; CHECK-NEXT: mov al, byte ptr [ebx + 11]
; CHECK-NEXT: mov al, byte ptr [ebx + 11]
; CHECK-NEXT: mov al, byte ptr [ebx + 11]

t6:
mov al, t2.FOOBAR.f.h
mov al, [t2].FOOBAR.f.h
mov al, [t2.FOOBAR].f.h
mov al, [t2.FOOBAR.f.h]

; CHECK: t6:
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]
; CHECK-NEXT: mov al, byte ptr [rip + t2+11]

t7:
mov al, [ebx].FOOBAR.e.b
mov al, [ebx.FOOBAR].e.b
mov al, [ebx.FOOBAR.e].b
mov al, [ebx.FOOBAR.e.b]

; CHECK: t7:
; CHECK-NEXT: mov al, byte ptr [ebx + 9]
; CHECK-NEXT: mov al, byte ptr [ebx + 9]
; CHECK-NEXT: mov al, byte ptr [ebx + 9]
; CHECK-NEXT: mov al, byte ptr [ebx + 9]

t8:
mov al, t2.FOOBAR.e.b
mov al, [t2].FOOBAR.e.b
mov al, [t2.FOOBAR].e.b
mov al, [t2.FOOBAR.e].b
mov al, [t2.FOOBAR.e.b]

; CHECK: t8:
; CHECK-NEXT: mov al, byte ptr [rip + t2+9]
; CHECK-NEXT: mov al, byte ptr [rip + t2+9]
; CHECK-NEXT: mov al, byte ptr [rip + t2+9]
; CHECK-NEXT: mov al, byte ptr [rip + t2+9]
; CHECK-NEXT: mov al, byte ptr [rip + t2+9]

QUUX STRUCT
  u DWORD ?
  UNION
    v WORD ?
    w DWORD ?
    STRUCT
      x BYTE ?
      y BYTE ?
    ENDS
    after_struct BYTE ?
  ENDS
  z DWORD ?
QUUX ENDS

t9:
mov eax, [ebx].QUUX.u
mov ax, [ebx].QUUX.v
mov eax, [ebx].QUUX.w
mov al, [ebx].QUUX.x
mov al, [ebx].QUUX.y
mov al, [ebx].QUUX.after_struct
mov eax, [ebx].QUUX.z

; CHECK: t9:
; CHECK-NEXT: mov eax, dword ptr [ebx]
; CHECK-NEXT: mov ax, word ptr [ebx + 4]
; CHECK-NEXT: mov eax, dword ptr [ebx + 4]
; CHECK-NEXT: mov al, byte ptr [ebx + 4]
; CHECK-NEXT: mov al, byte ptr [ebx + 5]
; CHECK-NEXT: mov al, byte ptr [ebx + 4]
; CHECK-NEXT: mov eax, dword ptr [ebx + 8]

t10:
mov eax, FOOBAR.f
mov eax, FOOBAR.f.h

; CHECK: t10:
; CHECK-NEXT: mov eax, 10
; CHECK-NEXT: mov eax, 11

t11:
mov ax, (FOOBAR PTR [ebx]).f
mov ax, (FOOBAR PTR t1).f

; CHECK: t11:
; CHECK-NEXT: mov ax, word ptr [ebx + 10]
; CHECK-NEXT: mov ax, word ptr [rip + t1+10]

END
