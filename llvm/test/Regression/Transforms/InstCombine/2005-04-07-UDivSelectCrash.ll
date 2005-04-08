; RUN: llvm-as < %s | opt -instcombine -disable-output

uint %test(bool %C, uint %tmp.15) {
	%tmp.16 = select bool %C, uint 8, uint 1
	%tmp.18 = div uint %tmp.15, %tmp.16
	ret uint %tmp.18
}
