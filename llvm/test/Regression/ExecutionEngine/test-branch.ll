; RUN: llvm-as -f %s -o %t.bc
; RUN: lli %t.bc > /dev/null

; test unconditional branch
int %main() {
	br label %Test
Test:
	%X = seteq int 0, 4
	br bool %X, label %Test, label %Label
Label:
	ret int 0
}
