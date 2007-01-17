; fsqrt should be generated when the fsqrt feature is enabled, but not 
; otherwise.

; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=+fsqrt | grep 'fsqrt f1, f1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mcpu=g5      | grep 'fsqrt f1, f1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mattr=-fsqrt | not grep 'fsqrt f1, f1' &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=ppc32 -mtriple=powerpc-apple-darwin8 -mcpu=g4 | not grep 'fsqrt f1, f1'

declare double %llvm.sqrt.f64(double)
double %X(double %Y) {
	%Z = call double %llvm.sqrt.f64(double %Y)
	ret double %Z
}
