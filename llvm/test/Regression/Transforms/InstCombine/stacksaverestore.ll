; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep call

;; Test that llvm.stackrestore is removed when possible.

int* %test1(uint %P) {
        %tmp = call sbyte* %llvm.stacksave()
        call void %llvm.stackrestore(sbyte* %tmp) ;; not restoring anything
	%A = alloca int, uint %P
        ret int* %A
}

void %test2(sbyte* %X) {
	call void %llvm.stackrestore(sbyte* %X)  ;; no allocas before return.
	ret void
}

declare sbyte* %llvm.stacksave()

declare void %llvm.stackrestore(sbyte*)
