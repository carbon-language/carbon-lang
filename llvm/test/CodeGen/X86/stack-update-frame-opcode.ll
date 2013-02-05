; RUN: llc -mtriple=x86_64-pc-linux < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-linux-gnux32 < %s | FileCheck -check-prefix=X32ABI %s

define i32 @bar(i32 %a) nounwind {
entry:
  %arr = alloca [400 x i32], align 16

; In the x32 ABI, the stack pointer is treated as a 32-bit value.
; CHECK: subq $1608
; X32ABI: subl $1608

  %arraydecay = getelementptr inbounds [400 x i32]* %arr, i64 0, i64 0
  %call = call i32 @foo(i32 %a, i32* %arraydecay) nounwind
  ret i32 %call

; CHECK: addq $1608
; X32ABI: addl $1608

}

declare i32 @foo(i32, i32*)

