; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep zapnot

ulong %foo(ulong %y) {
        %tmp = shl ulong %y, ubyte 3            ; <ulong> [#uses=1]
        %tmp2 = and ulong %tmp, 65535            ; <ulong> [#uses=1]
        ret ulong %tmp2
}

