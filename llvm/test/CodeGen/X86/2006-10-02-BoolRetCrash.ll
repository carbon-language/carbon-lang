; RUN: llvm-upgrade < %s | llvm-as | llc 
; PR933

fastcc bool %test() {
	ret bool true
}
