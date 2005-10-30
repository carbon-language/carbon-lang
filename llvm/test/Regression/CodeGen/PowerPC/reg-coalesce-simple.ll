; RUN: llvm-as < %s | llc -march=ppc32 &&
; RUN: llvm-as < %s | llc -march=ppc32 | not grep or

%struct.foo = type { int, int, [0 x ubyte] }
int %test(%struct.foo* %X) {
        %tmp1 = getelementptr %struct.foo* %X, int 0, uint 2, int 100
        %tmp = load ubyte* %tmp1                ; <ubyte> [#uses=1]
        %tmp2 = cast ubyte %tmp to int          ; <int> [#uses=1]
        ret int %tmp2}



