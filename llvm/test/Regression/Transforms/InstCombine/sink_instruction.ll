; RUN: llvm-as < %s | opt -instcombine | llvm-dis | %prcontext div 1 | grep ret

;; This tests that the instructions in the entry blocks are sunk into each
;; arm of the 'if'.

int %foo(bool %C, int %A, int %B) {
entry:
	%tmp.2 = div int %A, %B
	%tmp.9 = add int %B, %A
	br bool %C, label %then, label %endif

then:
	ret int %tmp.9

endif:
	ret int %tmp.2
}
