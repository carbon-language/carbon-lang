; RUN: llc -mcpu=pwr7 -O0 -relocation-model=pic < %s | FileCheck %s

; Test correct assembly code generation for thread-local storage using
; the general dynamic model.

target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

@a = thread_local global i32 0, align 4

define signext i32 @main() nounwind {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @a, align 4
  ret i32 %0
}

; CHECK: addis [[REG:[0-9]+]], 2, a@got@tlsgd@ha
; CHECK-NEXT: addi 3, [[REG]], a@got@tlsgd@l
; CHECK:      bl __tls_get_addr(a@TLSGD)
; CHECK-NEXT: nop

