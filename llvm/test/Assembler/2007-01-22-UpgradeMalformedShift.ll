; Found by inspection of the code
; RUN: llvm-upgrade < %s > /dev/null |& grep {Shift constant expression}

global int shr (float 1.0, ubyte 2)
