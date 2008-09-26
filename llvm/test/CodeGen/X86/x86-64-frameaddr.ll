; RUN: llvm-as < %s | llc -march=x86-64 | grep {leaq	-8(%rsp), %rax}
@llvm.noinline = appending global [1 x i8*] [ i8* bitcast (i64* ()* @stack_end_address to i8*) ], section "llvm.metadata"

define internal i64* @stack_end_address() nounwind  {
entry:
	tail call i8* @llvm.frameaddress( i32 0 )
	bitcast i8* %0 to i64*
	ret i64* %1
}

declare i8* @llvm.frameaddress(i32) nounwind readnone 
