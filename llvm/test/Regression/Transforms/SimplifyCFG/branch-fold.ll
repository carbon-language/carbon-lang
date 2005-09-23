; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep 'br bool' | wc -l | grep 1

void %test(int* %P, int* %Q, bool %A, bool %B) {
	br bool %A, label %a, label %b   ;; fold the two branches into one
a:
	br bool %B, label %b, label %c
b:
	store int 123, int* %P
	ret void
c:
	ret void
}
