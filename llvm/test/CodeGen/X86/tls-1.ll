; RUN: llc < %s -mtriple x86_64-apple-darwin | FileCheck %s

@a = thread_local global i32 0                    ; <i32*> [#uses=0]
@b = thread_local global i32 0                    ; <i32*> [#uses=0]

; CHECK: .tbss _a$tlv$init, 4, 2
; CHECK:        .section        __DATA,__thread_vars,thread_local_variables
; CHECK:        .globl  _a
; CHECK: _a:
; CHECK:        .quad   __tlv_bootstrap
; CHECK:        .quad   0
; CHECK:        .quad   _a$tlv$init

; CHECK: .tbss _b$tlv$init, 4, 2
; CHECK:        .globl  _b
; CHECK: _b:
; CHECK:        .quad   __tlv_bootstrap
; CHECK:        .quad   0
; CHECK:        .quad   _b$tlv$init
