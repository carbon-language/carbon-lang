; This testcase is incorrectly getting completely eliminated.  There should be
; SOME instruction named %c here, even if it's a bitwise and.
;
; RUN: as < %s | opt -instcombine | grep '%c'
ulong %test3(ulong %A) {
        %c1 = cast ulong %A to ubyte            ; <ubyte> [#uses=0]
        %c2 = cast ulong %A to ulong            ; <ulong> [#uses=0]
        ret ulong %A
}

