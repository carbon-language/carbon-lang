; RUN: llc < %s -mtriple=arm-apple-darwin | FileCheck %s -check-prefix=INSTR
; RUN: llc < %s -mtriple=arm-apple-darwin -trap-func=_trap | FileCheck %s -check-prefix=FUNC
; RUN: llc < %s -mtriple=arm-apple-darwin -trap-func=_trap -O0 | FileCheck %s -check-prefix=FUNC
; RUN: llc -mtriple=armv7-unknown-nacl -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7-unknown-nacl - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL
; RUN: llc -mtriple=armv7-unknown-nacl -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7 -mattr=+nacl-trap - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL
; RUN: llc -mtriple=armv7 -mattr=+nacl-trap -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7 -mattr=+nacl-trap - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL
; RUN: llc -verify-machineinstrs -fast-isel -mtriple=armv7-unknown-nacl -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7-unknown-nacl - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-NACL
; RUN: llc -mtriple=armv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-ALL
; RUN: llc -verify-machineinstrs -fast-isel -mtriple=armv7 -filetype=obj %s -o - \
; RUN:  | llvm-objdump -disassemble -triple armv7 - \
; RUN:  | FileCheck %s -check-prefix=ENCODING-ALL
; rdar://7961298
; rdar://9249183

define void @t() nounwind {
entry:
; INSTR-LABEL: t:
; INSTR: trap

; FUNC-LABEL: t:
; FUNC: bl __trap

; ENCODING-NACL: f0 de fe e7

; ENCODING-ALL: fe de ff e7

  call void @llvm.trap()
  unreachable
}

define void @t2() nounwind {
entry:
; INSTR-LABEL: t2:
; INSTR: trap

; FUNC-LABEL: t2:
; FUNC: bl __trap

; ENCODING-NACL: f0 de fe e7

; ENCODING-ALL: fe de ff e7

  call void @llvm.debugtrap()
  unreachable
}

declare void @llvm.trap() nounwind
declare void @llvm.debugtrap() nounwind
