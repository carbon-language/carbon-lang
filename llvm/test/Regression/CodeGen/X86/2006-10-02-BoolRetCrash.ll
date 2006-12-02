; RUN: llvm-upgrade < %s | llvm-as | llc &&
; RUN: llvm-upgrade < %s | llvm-as | llc -enable-x86-fastcc
; PR933

fastcc bool %test() {
	ret bool true
}
