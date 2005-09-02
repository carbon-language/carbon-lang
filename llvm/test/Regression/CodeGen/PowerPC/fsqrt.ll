; fsqrt should be generated when the fsqrt feature is enabled, but not 
; otherwise.

; RUN: llvm-as < %s | llc -march=ppc32 -mattr=+fsqrt | grep 'fsqrt f1, f1' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5      | grep 'fsqrt f1, f1' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mattr=-fsqrt | not grep 'fsqrt f1, f1' &&
; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g4 | not grep 'fsqrt f1, f1'

declare double %llvm.sqrt(double)
double %X(double %Y) {
	%Z = call double %llvm.sqrt(double %Y)
	ret double %Z
}
