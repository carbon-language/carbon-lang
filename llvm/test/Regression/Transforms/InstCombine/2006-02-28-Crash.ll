; RUN: llvm-upgrade < %s | llvm-as | opt -instcombine -disable-output
int %test() {
        %tmp203 = seteq uint 1, 2               ; <bool> [#uses=1]
        %tmp203 = cast bool %tmp203 to int              ; <int> [#uses=1]
	ret int %tmp203
}
