; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -enable-ppc-preinc | \
; RUN:   not grep addi
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc64 -enable-ppc-preinc | \
; RUN:   not grep addi
%Glob = global ulong 4

int *%test0(int *%X,  int *%dest) {
	%Y = getelementptr int* %X, int 4
	%A = load int* %Y
	store int %A, int* %dest
	ret int* %Y
}

int *%test1(int *%X,  int *%dest) {
	%Y = getelementptr int* %X, int 4
	%A = load int* %Y
	store int %A, int* %dest
	ret int* %Y
}

short *%test2(short *%X, int *%dest) {
	%Y = getelementptr short* %X, int 4
	%A = load short* %Y
	%B = cast short %A to int
	store int %B, int* %dest
	ret short* %Y
}

ushort *%test3(ushort *%X, int *%dest) {
	%Y = getelementptr ushort* %X, int 4
	%A = load ushort* %Y
	%B = cast ushort %A to int
	store int %B, int* %dest
	ret ushort* %Y
}

short *%test3a(short *%X, long *%dest) {
	%Y = getelementptr short* %X, int 4
	%A = load short* %Y
	%B = cast short %A to long
	store long %B, long* %dest
	ret short* %Y
}

long *%test4(long *%X, long *%dest) {
	%Y = getelementptr long* %X, int 4
	%A = load long* %Y
	store long %A, long* %dest
	ret long* %Y
}

ushort *%test5(ushort *%X) {
	%Y = getelementptr ushort* %X, int 4
	store ushort 7, ushort* %Y
	ret ushort* %Y
}

ulong *%test6(ulong *%X, ulong %A) {
	%Y = getelementptr ulong* %X, int 4
	store ulong %A, ulong* %Y
	ret ulong* %Y
}

ulong *%test7(ulong *%X, ulong %A) {
	store ulong %A, ulong* %Glob
	ret ulong *%Glob
}

