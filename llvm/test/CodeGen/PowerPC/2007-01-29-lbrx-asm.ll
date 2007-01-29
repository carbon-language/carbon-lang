; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc64

define i16 @test(i8* %data, i16* %data) {
	%tmp237 = call i16 asm "lhbrx $0, $2, $1", "=r,r,bO,m"( i8* %data, i32 0, i16* %data )		; <i16> [#uses=1]
	ret i16 %tmp237
}
