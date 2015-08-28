; RUN: llc -mtriple=x86_64-unknown-linux-gnu -o - %s | FileCheck %s

define void @f(i32** %p) unnamed_addr {
entry:
  %v = tail call i32* @g()
  store i32* %v, i32** %p, align 8
  ret void
}
; CHECK-LABEL: f:
; CHECK: callq g
; CHECK: movq    %rax, (%rbx)

declare i32* @g() #2

attributes #2 = { nounwind readnone }
