; RUN: llvm-as < %s | llc -march=x86-64 | FileCheck %s -check-prefix=CHECK-64
; RUN: llvm-as < %s | llc -march=x86 -mtriple=i686-apple-darwin9 -relocation-model=pic | FileCheck %s -check-prefix=CHECK-32
; PR3379

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@per_cpu__cpu_number = external global i32              ; <i32*> [#uses=1]

declare void @bar(...)

define i32 @test1() nounwind {
entry:
; Should have a rip suffix.
; CHECK-64: test1:
; CHECK-64: movl %gs:per_cpu__cpu_number(%rip),%eax

; CHECK-32: test1:
; CHECK-32: movl %gs:(%eax),%eax
        %A = call i32 asm "movl %gs:$1,$0",
            "=r,*m"(i32* @per_cpu__cpu_number) nounwind
        ret i32 %A
}

define i32 @test2() nounwind {
entry:
; Should not have a rip suffix because of the P modifier.
; CHECK-64: test2:
; CHECK-64: movl %gs:per_cpu__cpu_number,%eax

; CHECK-32: test2:
; CHECK-32: movl %gs:(%eax),%eax

        %A = call i32 asm "movl %gs:${1:P},$0",
            "=r,*m"(i32* @per_cpu__cpu_number) nounwind
        ret i32 %A
}

define void @test3() nounwind {
entry:
; CHECK-64: test3:
; CHECK-64: call bar
; CHECK-64: call test3

; CHECK-32: test3:
; CHECK-32: call _bar
; CHECK-32: call _test3
  tail call void asm sideeffect "call ${0:P}", "X"(void (...)* @bar) nounwind
  tail call void asm sideeffect "call ${0:P}", "X"(void (...)* bitcast (void ()* @test3 to void (...)*)) nounwind
  ret void
}
