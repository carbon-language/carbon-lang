; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux-gnu -relocation-model=pic  > %t
; RUN: grep {leal	i@TLSGD(,%ebx,1), %eax} %t
; RUN: grep {call	___tls_get_addr@PLT} %t

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}
