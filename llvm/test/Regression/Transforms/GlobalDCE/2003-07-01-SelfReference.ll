; distilled from 255.vortex
; RUN: llvm-upgrade < %s | llvm-as | opt -globaldce | llvm-dis | not grep testfunc

implementation

declare bool()* %getfunc()
internal bool %testfunc() {
	%F = call bool()*()* %getfunc()
	%c = seteq bool()* %F, %testfunc
	ret bool %c
}
