; RUN: not llvm-as %s -o /dev/null -f |& grep \
; RUN:   "Redefinition of global variable named 'B'"
; END.

@B = global i32 7
@B = global i32 7
