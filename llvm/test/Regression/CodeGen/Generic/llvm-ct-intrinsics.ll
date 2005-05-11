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

declare long %llvm.ctlz(long)
declare int %llvm.ctlz(int)
declare short %llvm.ctlz(short)
declare sbyte %llvm.ctlz(sbyte)

void %ctlztest(sbyte %A, short %B, int %C, long %D, 
               sbyte *%AP, short* %BP, int* %CP, long* %DP) {
	%a = call sbyte %llvm.ctlz(sbyte %A)
	%b = call short %llvm.ctlz(short %B)
	%c = call int %llvm.ctlz(int %C)
	%d = call long %llvm.ctlz(long %D)

	store sbyte %a, sbyte* %AP
	store short %b, short* %BP
	store int   %c, int* %CP
	store long  %d, long* %DP
	ret void
}

declare long %llvm.cttz(long)
declare int %llvm.cttz(int)
declare short %llvm.cttz(short)
declare sbyte %llvm.cttz(sbyte)

void %cttztest(sbyte %A, short %B, int %C, long %D, 
               sbyte *%AP, short* %BP, int* %CP, long* %DP) {
	%a = call sbyte %llvm.cttz(sbyte %A)
	%b = call short %llvm.cttz(short %B)
	%c = call int %llvm.cttz(int %C)
	%d = call long %llvm.cttz(long %D)

	store sbyte %a, sbyte* %AP
	store short %b, short* %BP
	store int   %c, int* %CP
	store long  %d, long* %DP
	ret void
}
