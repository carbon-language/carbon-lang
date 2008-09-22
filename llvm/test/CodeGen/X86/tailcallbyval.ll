; RUN: llvm-as < %s | llc -march=x86 -tailcallopt | grep TAILCALL
; check for the 2 byval moves
; RUN: llvm-as < %s | llc -march=x86 -tailcallopt | grep movl | grep ecx | grep eax | wc -l | grep 1
%struct.s = type {i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32,
                  i32, i32, i32, i32, i32, i32, i32, i32 }

define  fastcc i32 @tailcallee(%struct.s* byval %a) {
entry:
        %tmp2 = getelementptr %struct.s* %a, i32 0, i32 0
        %tmp3 = load i32* %tmp2
        ret i32 %tmp3
}

define  fastcc i32 @tailcaller(%struct.s* byval %a) {
entry:
        %tmp4 = tail call fastcc i32 @tailcallee(%struct.s* %a byval)
        ret i32 %tmp4
}
