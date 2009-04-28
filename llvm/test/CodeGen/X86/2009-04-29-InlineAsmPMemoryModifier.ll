; RUN: llvm-as < %s | llc -march=x86-64 | grep gs: | not grep rip
; PR3379

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"
@per_cpu__cpu_number = external global i32              ; <i32*> [#uses=1]

define void @pat_init() nounwind {
entry:
        %0 = call i32 asm "movl %gs:${1:P},$0", "=r,*m,~{dirflag},~{fpsr},~{flags}"(i32* @per_cpu__cpu_number) nounwind         ; <i32> [#uses=0]
        unreachable
}
