; RUN: llvm-as < %s | llc -mtriple=i386-apple-darwin9 -O0 | grep {movl	%edx, 12(%esp)} | count 2
; rdar://6992609

target triple = "i386-apple-darwin9.0"
@llvm.used = appending global [1 x i8*] [i8* bitcast (i64 (i64)* @_OSSwapInt64 to i8*)], section "llvm.metadata"		; <[1 x i8*]*> [#uses=0]

define i64 @_OSSwapInt64(i64 %_data) nounwind {
entry:
	%retval = alloca i64		; <i64*> [#uses=2]
	%_data.addr = alloca i64		; <i64*> [#uses=4]
	store i64 %_data, i64* %_data.addr
	%tmp = load i64* %_data.addr		; <i64> [#uses=1]
	%0 = call i64 asm "bswap   %eax\0A\09bswap   %edx\0A\09xchgl   %eax, %edx", "=A,0,~{dirflag},~{fpsr},~{flags}"(i64 %tmp) nounwind		; <i64> [#uses=1]
	store i64 %0, i64* %_data.addr
	%tmp1 = load i64* %_data.addr		; <i64> [#uses=1]
	store i64 %tmp1, i64* %retval
	%1 = load i64* %retval		; <i64> [#uses=1]
	ret i64 %1
}
