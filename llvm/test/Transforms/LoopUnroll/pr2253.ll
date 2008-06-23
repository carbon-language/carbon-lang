; RUN: llvm-as < %s | opt -loop-unroll -unroll-count 2 | llvm-dis | grep add | count 2
; PR2253

; There's a use outside the loop, and the PHI needs an incoming edge for
; each unrolled iteration, since the trip count is unknown and any iteration
; could exit.

define i32 @fib(i32 %n) nounwind {
entry:
        br i1 false, label %bb, label %return

bb:
        %t0 = phi i32 [ 0, %entry ], [ %t1, %bb ]
        %t1 = add i32 %t0, 1
        %c = icmp ne i32 %t0, %n
        br i1 %c, label %bb, label %return

return:
        %f2.0.lcssa = phi i32 [ -1, %entry ], [ %t0, %bb ]
        ret i32 %f2.0.lcssa
}
