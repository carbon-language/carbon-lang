; RUN: llc < %s -regalloc=local -relocation-model=pic | FileCheck %s

target triple = "thumbv6-apple-darwin10"

@fred = internal global i32 0              ; <i32*> [#uses=1]

define arm_apcscc void @foo() nounwind {
entry:
; CHECK: str r0, [sp]
  %0 = call arm_apcscc  i32 (...)* @bar() nounwind ; <i32> [#uses=1]
; CHECK: blx _bar
; CHECK: ldr r1, [sp]
  store i32 %0, i32* @fred, align 4
  br label %return

return:                                           ; preds = %entry
  ret void
}

declare arm_apcscc i32 @bar(...)
