; The assembler should catch an undefined argument type .
; RUN: llvm-as < %s -o /dev/null -f 2>&1 | grep "Reference to abstract argument"

; %typedef.bc_struct = type opaque

implementation   ; Functions:

define bool %someFunc(i32* %tmp.71.reload, %typedef.bc_struct* %n1) {
	ret bool true
}
