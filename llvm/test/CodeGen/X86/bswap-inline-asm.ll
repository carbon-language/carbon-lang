; RUN: llc < %s -march=x86-64 > %t
; RUN: not grep APP %t
; RUN: grep bswapq %t | count 2
; RUN: grep bswapl %t | count 1

define i64 @foo(i64 %x) nounwind {
	%asmtmp = tail call i64 asm "bswap $0", "=r,0,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
	ret i64 %asmtmp
}
define i64 @bar(i64 %x) nounwind {
	%asmtmp = tail call i64 asm "bswapq ${0:q}", "=r,0,~{dirflag},~{fpsr},~{flags}"(i64 %x) nounwind
	ret i64 %asmtmp
}
define i32 @pen(i32 %x) nounwind {
	%asmtmp = tail call i32 asm "bswapl ${0:q}", "=r,0,~{dirflag},~{fpsr},~{flags}"(i32 %x) nounwind
	ret i32 %asmtmp
}
