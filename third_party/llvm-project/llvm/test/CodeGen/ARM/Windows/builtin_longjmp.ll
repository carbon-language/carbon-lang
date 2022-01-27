; RUN: llc -mtriple thumbv7--windows-itanium -filetype asm -o - %s | FileCheck %s

declare void @llvm.eh.sjlj.longjmp(i8*)

define arm_aapcs_vfpcc void @test___builtin_longjump(i8* %b) {
entry:
  tail call void @llvm.eh.sjlj.longjmp(i8* %b)
  unreachable
}

; CHECK: push.w  {r11, lr}
; CHECK: ldr.w   r11, [r0]
; CHECK: ldr.w   sp, [r0, #8]
; CHECK: ldr.w   pc, [r0, #4]

