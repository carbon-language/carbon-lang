; RUN: llc < %s -tailcallopt -march=x86-64 -post-RA-scheduler=true | FileCheck %s

; Check that lowered arguments on the stack do not overwrite each other.
; Add %in1 %p1 to a different temporary register (%eax).
; CHECK: movl  32(%rsp), %eax
; Move param %in1 to temp register (%r10d).
; CHECK: movl  40(%rsp), %r10d
; Add %in1 %p1 to a different temporary register (%eax).
; CHECK: addl %edi, %eax
; Move param %in2 to stack.
; CHECK: movl  %r10d, 32(%rsp)
; Move result of addition to stack.
; CHECK: movl  %eax, 40(%rsp)
; Eventually, do a TAILCALL
; CHECK: TAILCALL

declare fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %a, i32 %b) nounwind

define fastcc i32 @tailcaller(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %in1, i32 %in2) nounwind {
entry:
        %tmp = add i32 %in1, %p1
        %retval = tail call fastcc i32 @tailcallee(i32 %p1, i32 %p2, i32 %p3, i32 %p4, i32 %p5, i32 %p6, i32 %in2,i32 %tmp)
        ret i32 %retval
}

