; The loop sinker was running from the bottom of the loop to the top, causing
; it to miss opportunities to sink instructions that depended on sinking other
; instructions from the loop.  Instead they got hoisted, which is better than
; leaving them in the loop, but increases register pressure pointlessly.

; RUN: llvm-as < %s | opt -licm | llvm-dis | grep -C1 getelementptr | grep Out:

%Ty = type { int, int }
%X = external global %Ty

int %test() {
        br label %Loop
Loop:
        %dead = getelementptr %Ty* %X, long 0, ubyte 0
        %sunk2 = load int* %dead
        br bool false, label %Loop, label %Out
Out:
        ret int %sunk2
}
