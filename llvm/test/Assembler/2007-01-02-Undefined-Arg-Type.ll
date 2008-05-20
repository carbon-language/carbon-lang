; The assembler should catch an undefined argument type .
; RUN: not llvm-as %s -o /dev/null -f |& grep {Reference to abstract argument}

; %typedef.bc_struct = type opaque


define i1 @someFunc(i32* %tmp.71.reload, %typedef.bc_struct* %n1) {
	ret i1 true
}
