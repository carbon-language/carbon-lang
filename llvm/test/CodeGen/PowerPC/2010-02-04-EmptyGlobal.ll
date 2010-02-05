; RUN: llc < %s -mtriple=powerpc-apple-darwin10 -relocation-model=pic -disable-fp-elim | FileCheck %s
; <rdar://problem/7604010>

%cmd.type = type { }

@_cmd = constant %cmd.type zeroinitializer

; CHECK:      .globl __cmd
; CHECK-NEXT: .align 3
; CHECK-NEXT: __cmd:
; CHECK-NEXT: .space 1
