; Found by inspection of the code
; RUN: llvm-upgrade < %s  > /dev/null |& grep {Illegal number of init}

global {} { int 7, float 1.0, int 7, int 8 }
