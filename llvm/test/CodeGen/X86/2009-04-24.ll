; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu -regalloc=local -relocation-model=pic > %t
; RUN: grep {leal.*TLSGD.*___tls_get_addr} %t
; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-linux-gnu -regalloc=local -relocation-model=pic > %t2
; RUN: grep {leaq.*TLSGD.*__tls_get_addr} %t2
; PR/4004

@i = thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}
