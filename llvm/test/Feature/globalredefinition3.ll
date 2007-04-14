; RUN: ignore llvm-as < %s -o /dev/null -f |& \
; RUN:   grep "Redefinition of global variable named 'B'"
; END.

@B = global i32 7
@B = global i32 7
