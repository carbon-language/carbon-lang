; RUN: not llvm-ml -filetype=s %s /Fo /dev/null 2>&1 | FileCheck %s --dump-input=always

.data

FOO STRUCT
  dword_field DWORD 3
  byte_field BYTE 4 DUP (1)
FOO ENDS

var FOO <>

.code

t1 PROC

mov eax, var.byte_field
; CHECK: error: invalid operand for instruction

mov eax, [var].byte_field
; CHECK: error: invalid operand for instruction

mov eax, [var.byte_field]
; CHECK: error: invalid operand for instruction

t1 ENDP

END
