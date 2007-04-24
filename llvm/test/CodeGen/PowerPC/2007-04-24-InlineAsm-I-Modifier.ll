; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8.8.0 | grep {foo r3, r4}
; RUN: llvm-as < %s | llc -march=ppc32 -mtriple=powerpc-apple-darwin8.8.0 | grep {bar r3, r}

; PR1351

define i32 @test1(i32 %Y, i32 %X) {
	%tmp1 = tail call i32 asm "foo${1:I} $0, $1", "=r,rI"( i32 %X )
	ret i32 %tmp1
}

;; TODO: We'd actually prefer this to be 'bari r3, 47', but 'bar r3, rN' is also ok.
define i32 @test2(i32 %Y, i32 %X) {
	%tmp1 = tail call i32 asm "bar${1:I} $0, $1", "=r,rI"( i32 47 )
	ret i32 %tmp1
}
