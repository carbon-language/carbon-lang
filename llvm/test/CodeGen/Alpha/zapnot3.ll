; RUN: llvm-upgrade < %s | llvm-as | llc -march=alpha | grep zapnot

;demanded bits mess up this mask in a hard to fix way
;ulong %foo(ulong %y) {
;        %tmp = and ulong %y,  65535
;        %tmp2 = shr ulong %tmp,  ubyte 3
;        ret ulong %tmp2
;}

ulong %foo2(ulong %y) {
        %tmp = shr ulong %y, ubyte 3            ; <ulong> [#uses=1]
        %tmp2 = and ulong %tmp, 8191            ; <ulong> [#uses=1]
        ret ulong %tmp2
}

