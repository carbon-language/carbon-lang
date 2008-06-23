; RUN: llvm-as < %s | opt -loop-unroll -unroll-count 6 -unroll-threshold 300 | llvm-dis > %t
; RUN: grep {br label \%bbe} %t | count 12
; RUN: grep {br i1 \%z} %t | count 3
; RUN: grep {br i1 \%q} %t | count 6
; RUN: grep call %t | count 12
; RUN: grep urem %t | count 6
; RUN: grep store %t | count 6
; RUN: grep phi %t | count 11
; RUN: grep {lcssa = phi} %t | count 2

; This testcase uses
;  - an unknown tripcount, but a known trip multiple of 2.
;  - an unroll count of 6, so we should get 3 conditional branches
;    in the loop.
;  - values defined inside the loop and used outside, by phis that
;    also use values defined elsewhere outside the loop.
;  - a phi inside the loop that only uses values defined
;    inside the loop and is only used inside the loop.

declare i32 @foo()
declare i32 @bar()

define i32 @fib(i32 %n, i1 %a, i32* %p) nounwind {
entry:
        %n2 = mul i32 %n, 2
        br i1 %a, label %bb, label %return

bb: ; loop header block
        %t0 = phi i32 [ 0, %entry ], [ %t1, %bbe ]
        %td = urem i32 %t0, 7
        %q = trunc i32 %td to i1
        br i1 %q, label %bbt, label %bbf
bbt:
        %bbtv = call i32 @foo()
        br label %bbe
bbf:
        %bbfv = call i32 @bar()
        br label %bbe
bbe: ; loop latch block
        %bbpv = phi i32 [ %bbtv, %bbt ], [ %bbfv, %bbf ]
        store i32 %bbpv, i32* %p
        %t1 = add i32 %t0, 1
        %z = icmp ne i32 %t1, %n2
        br i1 %z, label %bb, label %return

return:
        %f = phi i32 [ -2, %entry ], [ %t0, %bbe ]
        %g = phi i32 [ -3, %entry ], [ %t1, %bbe ]
        %h = mul i32 %f, %g
        ret i32 %h
}
