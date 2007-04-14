; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine | llvm-dis | \
; RUN:   %prcontext div 1 | grep then:

;; This tests that the div is hoisted into the then block.

int %foo(bool %C, int %A, int %B) {
entry:
	br bool %C, label %then, label %endif

then:
	br label %endif

endif:
	%X = phi int [%A, %then], [15, %entry]
	%Y = div int %X, 42
	ret int %Y
}
