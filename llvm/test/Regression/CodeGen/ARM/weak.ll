; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm &&
; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm | grep .weak

implementation   ; Functions:

weak uint %f() {
entry:
	unreachable
}
