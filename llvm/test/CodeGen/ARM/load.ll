; RUN: llvm-upgrade < %s | llvm-as | llc -march=arm > %t
; RUN: grep ldrsb %t
; RUN: grep ldrb %t
; RUN: grep ldrsh %t
; RUN: grep ldrh %t

int %f1(sbyte* %p) {
entry:
	%tmp = load sbyte* %p		; <sbyte> [#uses=1]
	%tmp = cast sbyte %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f2(ubyte* %p) {
entry:
	%tmp = load ubyte* %p		; <sbyte> [#uses=1]
	%tmp = cast ubyte %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f3(short* %p) {
entry:
	%tmp = load short* %p		; <sbyte> [#uses=1]
	%tmp = cast short %tmp to int		; <int> [#uses=1]
	ret int %tmp
}

int %f4(ushort* %p) {
entry:
	%tmp = load ushort* %p		; <sbyte> [#uses=1]
	%tmp = cast ushort %tmp to int		; <int> [#uses=1]
	ret int %tmp
}
