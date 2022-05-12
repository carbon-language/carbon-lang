; RUN: llc -fast-isel -O0 -code-model=large -mcpu=generic -mtriple=x86_64-linux -relocation-model=static < %s | FileCheck %s

; Check that fast-isel cleans up when it fails to lower a call instruction.
define void @fastiselcall() {
entry:
  %call = call i32 @targetfn(i32 42)
  ret void
; CHECK-LABEL: fastiselcall:
; FastISel's local value code was dead, so it's gone.
; CHECK-NOT: movl $42,
; SDag-ISel's arg mov:
; CHECK: movabsq $targetfn, %[[REG:[^ ]*]]
; CHECK: movl $42, %edi
; CHECK: callq *%[[REG]]

}
declare i32 @targetfn(i32)
