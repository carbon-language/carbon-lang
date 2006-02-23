; RUN: llvm-as < %s | llc -march=x86 | grep lea

%G = weak global int 0
int %test1(int* %P, int %X) {
	%tmp.1 = getelementptr int* %P, int %X
	%tmp.2 = load int* %tmp.1
        store int %tmp.2, int* %G
	%tmp.3 = sub int %tmp.2, 9
	ret int %tmp.3
}
