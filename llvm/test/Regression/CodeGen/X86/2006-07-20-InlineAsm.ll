; RUN: llvm-as < %s | llc -march=x86
; PR833

%G = weak global int 0		; <int*> [#uses=3]

implementation   ; Functions:

int %foo(int %X) {
entry:
	%X_addr = alloca int		; <int*> [#uses=3]
	store int %X, int* %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "==m,==r,m,1,~{dirflag},~{fpsr},~{flags}"( int* %G, int* %X_addr, int* %G, int %X )
	%tmp1 = load int* %X_addr		; <int> [#uses=1]
	ret int %tmp1
}

int %foo2(int %X) {
entry:
	%X_addr = alloca int		; <int*> [#uses=3]
	store int %X, int* %X_addr
	call void asm sideeffect "xchg{l} {$0,$1|$1,$0}", "==m,==r,1,~{dirflag},~{fpsr},~{flags}"( int* %G, int* %X_addr, int %X )
	%tmp1 = load int* %X_addr		; <int> [#uses=1]
	ret int %tmp1
}
