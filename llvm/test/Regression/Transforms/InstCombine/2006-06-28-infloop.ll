; RUN: llvm-as < %s | opt -instcombine -disable-output
target endian = big
target pointersize = 32
target triple = "powerpc-apple-darwin8"

implementation   ; Functions:

void %test() {
entry:
	%tmp = getelementptr { long, long, long, long }* null, int 0, uint 3
	%tmp = load long* %tmp		; <long> [#uses=1]
	%tmp8 = load ulong* null		; <ulong> [#uses=1]
	%tmp8 = cast ulong %tmp8 to long		; <long> [#uses=1]
	%tmp9 = and long %tmp8, %tmp		; <long> [#uses=1]
	%sext = cast long %tmp9 to int		; <int> [#uses=1]
	%tmp27.i = cast int %sext to long		; <long> [#uses=1]
	tail call void %foo( uint 0, long %tmp27.i )
	unreachable
}

declare void %foo(uint, long)
