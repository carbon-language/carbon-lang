; Found by inspection of the code
; RUN: not llvm-upgrade < %s > /dev/null |& grep {Shift constant expression}

global int shr (float 1.0, ubyte 2)
