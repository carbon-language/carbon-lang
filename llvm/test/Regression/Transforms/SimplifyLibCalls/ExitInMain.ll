; Test that the ExitInMainOptimization pass works correctly
; RUN: llvm-as < %s | opt -simplify-libcalls | llvm-dis | grep -c 'ret int 3' | grep 1

declare void %exit(int)
declare void %exitonly(int)

implementation   ; Functions:

int %main () {
        call void %exitonly ( int 3 )
        call void %exit ( int 3 )
        ret int 0
}
