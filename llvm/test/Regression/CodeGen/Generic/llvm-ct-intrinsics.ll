; Make sure this testcase is supported by all code generators
; RUN: llvm-as < %s | llc

declare long %llvm.ctpop(long)
declare int %llvm.ctpop(int)
declare short %llvm.ctpop(short)
declare sbyte %llvm.ctpop(sbyte)

void %ctpoptest(sbyte %A, short %B, int %C, long %D, 
                sbyte *%AP, short* %BP, int* %CP, long* %DP) {
	%a = call sbyte %llvm.ctpop(sbyte %A)
	%b = call short %llvm.ctpop(short %B)
	%c = call int %llvm.ctpop(int %C)
	%d = call long %llvm.ctpop(long %D)

	store sbyte %a, sbyte* %AP
	store short %b, short* %BP
	store int   %c, int* %CP
	store long  %d, long* %DP
	ret void
}
