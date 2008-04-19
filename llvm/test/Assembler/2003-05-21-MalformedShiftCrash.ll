; Found by inspection of the code
; RUN: not llvm-as < %s > /dev/null |& grep {Logical operator requires integral}

global i32 ashr (float 1.0, float 2.0)
