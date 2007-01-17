; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 | grep xor 

target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8.7.0"

implementation   ; Functions:

void %foo(int %X) {
entry:
	%tmp1 = and int %X, 3		; <int> [#uses=1]
	%tmp2 = xor int %tmp1, 1
	%tmp = seteq int %tmp2, 0		; <bool> [#uses=1]
	br bool %tmp, label %UnifiedReturnBlock, label %cond_true

cond_true:		; preds = %entry
	tail call int (...)* %bar( )		; <int> [#uses=0]
	ret void

UnifiedReturnBlock:		; preds = %entry
	ret void
}

declare int %bar(...)
