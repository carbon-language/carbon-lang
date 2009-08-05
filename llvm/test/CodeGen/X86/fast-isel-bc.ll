; RUN: llvm-as < %s | llc -O0 -march=x86-64 -mattr=+mmx | FileCheck %s
; PR4684

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin9.8"

declare void @func2(<1 x i64>)

define void @func1() nounwind {

; This isn't spectacular, but it's MMX code at -O0...
; CHECK: movl $2, %eax
; CHECK: movd %rax, %mm0
; CHECK: movd %mm0, %rdi

        call void @func2(<1 x i64> <i64 2>)
        ret void
}
