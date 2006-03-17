; RUN: llvm-as < %s | llc -march=ppc32 | grep neg

int %test(int %X) {
    %Y = sub int 0, %X
	ret int %Y
}
