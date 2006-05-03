; RUN: llvm-as < %s | llc -march=x86 -relocation-model=static -stats 2>&1 | grep "asm-printer" | grep 14
%size20 = external global uint		; <uint*> [#uses=1]
%in5 = external global ubyte*		; <ubyte**> [#uses=1]

int %compare(sbyte* %a, sbyte* %b) {
	%tmp = cast sbyte* %a to uint*		; <uint*> [#uses=1]
	%tmp1 = cast sbyte* %b to uint*		; <uint*> [#uses=1]
	%tmp = load uint* %size20		; <uint> [#uses=1]
	%tmp = load ubyte** %in5		; <ubyte*> [#uses=2]
	%tmp3 = load uint* %tmp1		; <uint> [#uses=1]
	%tmp4 = getelementptr ubyte* %tmp, uint %tmp3		; <ubyte*> [#uses=1]
	%tmp7 = load uint* %tmp		; <uint> [#uses=1]
	%tmp8 = getelementptr ubyte* %tmp, uint %tmp7		; <ubyte*> [#uses=1]
	%tmp8 = cast ubyte* %tmp8 to sbyte*		; <sbyte*> [#uses=1]
	%tmp4 = cast ubyte* %tmp4 to sbyte*		; <sbyte*> [#uses=1]
	%tmp = tail call int %memcmp( sbyte* %tmp8, sbyte* %tmp4, uint %tmp )		; <int> [#uses=1]
	ret int %tmp
}

declare int %memcmp(sbyte*, sbyte*, uint)
