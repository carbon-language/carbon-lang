; Though this case seems to be fairly unlikely to occur in the wild, someone
; plunked it into the demo script, so maybe they care about it.
;
; RUN: llvm-as < %s | opt -tailcallelim | llvm-dis | not grep call

int %aaa(int %c) {
entry:
        %tmp.1 = seteq int %c, 0                ; <bool> [#uses=1]
        br bool %tmp.1, label %return, label %else

else:           ; preds = %entry
        %tmp.5 = add int %c, -1         ; <int> [#uses=1]
        %tmp.3 = call int %aaa( int %tmp.5 )            ; <int> [#uses=0]
        ret int 0

return:         ; preds = %entry
        ret int 0
}

