; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | not grep cmovlt
; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep cmoveq


long %cmov_lt(long %a, long %c) {
entry:
	%tmp.1 = setlt long %c, 0
	%retval = select bool %tmp.1, long %a, long 10
	ret long %retval
}

long %cmov_const(long %a, long %b, long %c) {
entry:
        %tmp.1 = setlt long %a, %b
        %retval = select bool %tmp.1, long %c, long 10
        ret long %retval
}

long %cmov_lt2(long %a, long %c) {
entry:
	%tmp.1 = setgt long %c, 0
	%retval = select bool %tmp.1, long 10, long %a
	ret long %retval
}
