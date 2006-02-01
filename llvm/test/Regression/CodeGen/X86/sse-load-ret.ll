; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -enable-x86-sse &&
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -enable-x86-sse | not grep movss
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah -enable-x86-sse | not grep xmm

double %test1(double *%P) {
	%X = load double* %P
	ret double %X
}

double %test2() {
	ret double 1234.56
}

; FIXME: Todo
;double %test3(bool %B) {
;	%C = select bool %B, double 123.412, double 523.01123123
;	ret double %C
;}

