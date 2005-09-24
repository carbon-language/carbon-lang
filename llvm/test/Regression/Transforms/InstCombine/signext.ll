; RUN: llvm-as < %s | opt -instcombine -disable-output &&
; RUN: llvm-as < %s | opt -instcombine | llvm-dis | not grep '(and\|xor\|add\|shl\|shr)'

int %test1(int %x) {
        %tmp.1 = and int %x, 65535              ; <int> [#uses=1]
        %tmp.2 = xor int %tmp.1, -32768         ; <int> [#uses=1]
        %tmp.3 = add int %tmp.2, 32768          ; <int> [#uses=1]
        ret int %tmp.3
}

int %test2(int %x) {
        %tmp.1 = and int %x, 65535              ; <int> [#uses=1]
        %tmp.2 = xor int %tmp.1, 32768          ; <int> [#uses=1]
        %tmp.3 = add int %tmp.2, -32768         ; <int> [#uses=1]
        ret int %tmp.3
}

int %test3(ushort %P) {
        %tmp.1 = cast ushort %P to int          ; <int> [#uses=1]
        %tmp.4 = xor int %tmp.1, 32768          ; <int> [#uses=1]
        %tmp.5 = add int %tmp.4, -32768         ; <int> [#uses=1]
        ret int %tmp.5
}

uint %test4(ushort %P) {
        %tmp.1 = cast ushort %P to uint         ; <uint> [#uses=1]
        %tmp.4 = xor uint %tmp.1, 32768         ; <uint> [#uses=1]
        %tmp.5 = add uint %tmp.4, 4294934528            ; <uint> [#uses=1]
        ret uint %tmp.5
}

int %test5(int %x) {
	%tmp.1 = and int %x, 254
	%tmp.2 = xor int %tmp.1, 128
	%tmp.3 = add int %tmp.2, -128
	ret int %tmp.3
}

int %test6(int %x) {
        %tmp.2 = shl int %x, ubyte 16           ; <int> [#uses=1]
        %tmp.4 = shr int %tmp.2, ubyte 16               ; <int> [#uses=1]
        ret int %tmp.4
}
