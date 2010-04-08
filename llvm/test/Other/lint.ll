; RUN: opt -lint -disable-output < %s |& FileCheck %s
target datalayout = "e-p:64:64:64"

declare fastcc void @bar()

define i32 @foo() noreturn {
; CHECK: Caller and callee calling convention differ
  call void @bar()
; CHECK: Null pointer dereference
  store i32 0, i32* null
; CHECK: Null pointer dereference
  %t = load i32* null
; CHECK: Memory reference address is misaligned
  %x = inttoptr i32 1 to i32*
  load i32* %x, align 4
; CHECK: Division by zero
  %sd = sdiv i32 2, 0
; CHECK: Division by zero
  %ud = udiv i32 2, 0
; CHECK: Division by zero
  %sr = srem i32 2, 0
; CHECK: Division by zero
  %ur = urem i32 2, 0
  br label %next

next:
; CHECK: Static alloca outside of entry block
  %a = alloca i32
; CHECK: Return statement in function with noreturn attribute
  ret i32 0
}
