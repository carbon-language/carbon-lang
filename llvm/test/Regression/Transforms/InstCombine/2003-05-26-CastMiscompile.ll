; RUN: llvm-as < %s | opt -instcombine | llvm-dis | grep 4294967295

ulong %test(ulong %Val) {
        %tmp.3 = cast ulong %Val to uint              ; <uint> [#uses=1]
        %tmp.8 = cast uint %tmp.3 to ulong              ; <ulong> [#uses=1]
	ret ulong %tmp.8
}
