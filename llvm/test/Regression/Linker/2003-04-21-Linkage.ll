; RUN: echo "%X = linkonce global int 5  implementation linkonce int %foo() { ret int 7 }" | as > Output/%s.1.bc
; RUN: as < %s > Output/%s.2.bc
; RUN: link Output/%s.[12].bc 
%X = external global int 

implementation

declare int %foo() 

void %bar() {
	load int* %X
	call int %foo()
	ret void
}

