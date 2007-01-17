; RUN: llvm-upgrade < %s | llvm-as | llc -march=x86 | grep test.*1
; PR1016

int %test(int %A, int %B, int %C) {
	%a = trunc int %A to bool
	%D = select bool %a, int %B, int %C
	ret int %D
}
