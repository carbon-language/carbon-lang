; RUN: llvm-as < %s | llc -march=c &&
; RUN: llvm-as < %s | llc -march=c | grep '\* *volatile *\*'

%G = external global void()*

void %test() {
	volatile store void()* %test, void()** %G
	volatile load void()** %G
	ret void
}
