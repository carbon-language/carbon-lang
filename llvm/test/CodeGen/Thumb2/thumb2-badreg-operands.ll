; RUN: llc < %s -mtriple=thumbv7-apple-darwin10 | FileCheck %s

define void @b(i32 %x) nounwind optsize {
entry:
; CHECK: b
; CHECK: mov r2, sp
; CHECK: mls r0, r0, r1, r2
; CHECK: mov sp, r0
  %0 = mul i32 %x, 24                             ; <i32> [#uses=1]
  %vla = alloca i8, i32 %0, align 1               ; <i8*> [#uses=1]
  call arm_aapcscc  void @a(i8* %vla) nounwind optsize
  ret void
}

declare void @a(i8*) optsize
