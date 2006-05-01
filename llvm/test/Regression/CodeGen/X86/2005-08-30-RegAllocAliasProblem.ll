; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'test.*AL' || \
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=intel | not grep 'cmove.*EAX'

; This testcase was compiling to:
;
; _pypy_simple5:
;        movl $13, %ecx
;        movl $12, %eax
;        movb 4(%esp), %al
;        testb %al, %al      ;; clobber EAX!
;        cmove %ecx, %eax
;        ret

int %pypy_simple5(bool %b_4787) {
	%retval = select bool %b_4787, int 12, int 13		; <int> [#uses=1]
	ret int %retval
}
