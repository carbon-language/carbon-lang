; RUN: llvm-as < %s | llc &&
; RUN: llvm-as < %s | llc -enable-x86-fastcc
; PR933

fastcc bool %test() {
	ret bool true
}
