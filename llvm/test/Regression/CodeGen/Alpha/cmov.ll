; RUN: llvm-as < %s | llc -march=alpha | not grep cmovlt
; RUN: llvm-as < %s | llc -march=alpha | grep cmoveq

long %cmovlt_(long %a, long %c) {
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

