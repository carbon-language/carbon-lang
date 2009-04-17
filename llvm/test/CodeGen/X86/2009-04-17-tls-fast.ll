; RUN: llvm-as < %s | llc -march=x86-64 -mtriple=x86_64-linux-gnu -relocation-model=pic -regalloc=local > %t
; RUN: grep {leaq	foo@TLSGD(%rip), %rdi} %t

@foo = internal thread_local global i32 100

define void @f(i32 %n) nounwind {
entry:
	%n_addr = alloca i32
	%p = alloca i32*
	%"alloca point" = bitcast i32 0 to i32
	store i32 %n, i32* %n_addr
	store i32* @foo, i32** %p, align 8
	br label %return

return:
	ret void
}
