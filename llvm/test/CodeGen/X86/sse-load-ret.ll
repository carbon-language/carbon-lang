; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=x86 -mcpu=yonah | not grep movss
; RUN: llvm-upgrade < %s | llvm-as | \
; RUN:   llc -march=x86 -mcpu=yonah | not grep xmm

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

