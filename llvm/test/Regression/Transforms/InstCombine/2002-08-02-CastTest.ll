; This testcase is incorrectly getting completely eliminated.  There should be
; SOME instruction named %c here, even if it's a bitwise and.
;
; RUN: llvm-as < %s | opt -instcombine -die | llvm-dis | grep '%c'
;
ulong %test3(ulong %A) {
        %c1 = cast ulong %A to ubyte
        %c2 = cast ubyte %c1 to ulong
        ret ulong %c2
}

