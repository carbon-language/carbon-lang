; RUN: llvm-as < %s | llc -march=x86 | not grep 'mov'
;
; Test the add and load are folded into the store instruction.

%X = weak global short 0

void %foo() {
	%tmp.0 = load short* %X
	%tmp.3 = add short %tmp.0, 329
	store short %tmp.3, short* %X
	ret void
}
