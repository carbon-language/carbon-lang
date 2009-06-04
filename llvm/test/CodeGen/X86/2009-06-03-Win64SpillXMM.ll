; RUN: llvm-as < %s | llc -o %t1 -f
; RUN: grep "subq.*\\\$40, \\\%rsp" %t1
; RUN: grep "movaps	\\\%xmm8, \\\(\\\%rsp\\\)" %t1
; RUN: grep "movaps	\\\%xmm7, 16\\\(\\\%rsp\\\)" %t1
target triple = "x86_64-mingw64"

define i32 @a() nounwind {
entry:
	tail call void asm sideeffect "", "~{xmm7},~{xmm8},~{dirflag},~{fpsr},~{flags}"() nounwind
	ret i32 undef
}

