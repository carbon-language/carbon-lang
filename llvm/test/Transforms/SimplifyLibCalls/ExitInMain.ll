; Test that the ExitInMainOptimization pass works correctly
; RUN: llvm-upgrade < %s | llvm-as | opt -simplify-libcalls | llvm-dis | \
; RUN:    grep {ret i32 3} | count 1
; END.

declare void %exit(int)
declare void %exitonly(int)

implementation   ; Functions:

int %main () {
        call void %exitonly ( int 3 )
        call void %exit ( int 3 )
        ret int 0
}
