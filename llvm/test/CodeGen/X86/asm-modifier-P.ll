; RUN: llvm-as < %s | llc -march=x86-64 | FileCheck %s
; PR3379

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@per_cpu__cpu_number = external global i32              ; <i32*> [#uses=1]

define void @test1() nounwind {
entry:
; Should have a rip suffix.
; CHECK: test1:
; CHECK: movl %gs:per_cpu__cpu_number(%rip),%eax
        %0 = call i32 asm "movl %gs:$1,$0",
            "=r,*m"(i32* @per_cpu__cpu_number) nounwind
        ret void
}

define void @test2() nounwind {
entry:
; Should not have a rip suffix because of the P modifier.
; CHECK: test2:
; CHECK: movl %gs:per_cpu__cpu_number,%eax
        %0 = call i32 asm "movl %gs:${1:P},$0",
            "=r,*m"(i32* @per_cpu__cpu_number) nounwind
        ret void
}

