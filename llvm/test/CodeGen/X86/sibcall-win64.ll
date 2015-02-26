; RUN: llc < %s -mtriple=x86_64-pc-linux | FileCheck %s

declare x86_64_win64cc void @win64_callee(i32)
declare void @sysv_callee(i32)

define void @sysv_caller(i32 %p1) {
entry:
  tail call x86_64_win64cc void @win64_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: sysv_caller:
; CHECK: subq $40, %rsp
; CHECK: callq win64_callee
; CHECK: addq $40, %rsp
; CHECK: retq

define x86_64_win64cc void @win64_caller(i32 %p1) {
entry:
  tail call void @sysv_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: win64_caller:
; CHECK: callq sysv_callee
; CHECK: retq

define void @sysv_matched(i32 %p1) {
  tail call void @sysv_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: sysv_matched:
; CHECK: jmp sysv_callee # TAILCALL

define x86_64_win64cc void @win64_matched(i32 %p1) {
  tail call x86_64_win64cc void @win64_callee(i32 %p1)
  ret void
}

; CHECK-LABEL: win64_matched:
; CHECK: jmp win64_callee # TAILCALL
