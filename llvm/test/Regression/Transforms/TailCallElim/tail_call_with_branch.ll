; This testcase is due to tail-duplication not wanting to copy the return
; instruction into the terminating blocks because there was other code
; optimized out of the function after the taildup happened.

; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

int %t4(int %a) {
entry:
        %tmp.1 = and int %a, 1
        %tmp.2 = cast int %tmp.1 to bool
        br bool %tmp.2, label %then.0, label %else.0

then.0:
        %tmp.5 = add int %a, -1
        %tmp.3 = call int %t4( int %tmp.5 )
        br label %return

else.0:
        %tmp.7 = setne int %a, 0
        br bool %tmp.7, label %then.1, label %return

then.1:
        %tmp.11 = add int %a, -2
        %tmp.9 = call int %t4( int %tmp.11 )
        br label %return

return:
        %result.0 = phi int [ 0, %else.0 ], [ %tmp.3, %then.0 ], [ %tmp.9, %then.1 ]
        ret int %result.0
}

