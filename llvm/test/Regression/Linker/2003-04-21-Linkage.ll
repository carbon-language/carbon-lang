; RUN: echo "%X = linkonce global int 5  implementation linkonce int %foo() { ret int 7 }" | llvm-as > %t.1.bc
; RUN: llvm-as < %s > %t.2.bc
; RUN: llvm-link %t.[12].bc 
%X = external global int 

implementation

declare int %foo() 

void %bar() {
	load int* %X
	call int %foo()
	ret void
}

