; A test for "physical subtyping" used in some C programs...
;
; RUN: llvm-upgrade < %s | llvm-as | opt -analyze -tddatastructure
;
%ST = type { int, int* }            ; "Subtype"
%DT = type { int, int*, int }       ; "derived type"

int %test(%DT* %DT) {
	%DTp = getelementptr %DT* %DT, long 0, uint 0
	%A = load int* %DTp
	%ST = cast %DT* %DT to %ST*
	%STp = getelementptr %ST* %ST, long 0, uint 0
	%B = load int* %STp
	%C = sub int %A, %B         ; A & B are equal, %C = 0
	ret int %C
}
