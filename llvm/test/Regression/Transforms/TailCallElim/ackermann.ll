; This function contains two tail calls, which should be eliminated
; RUN: llvm-as < %s | opt -tailcallelim -stats -disable-output 2>&1 | grep '2 tailcallelim'

int %Ack(int %M.1, int %N.1) {
entry:
        %tmp.1 = seteq int %M.1, 0              ; <bool> [#uses=1]
        br bool %tmp.1, label %then.0, label %endif.0

then.0:
        %tmp.4 = add int %N.1, 1                ; <int> [#uses=1]
        ret int %tmp.4

endif.0:
        %tmp.6 = seteq int %N.1, 0              ; <bool> [#uses=1]
        br bool %tmp.6, label %then.1, label %endif.1

then.1:
        %tmp.10 = add int %M.1, -1              ; <int> [#uses=1]
        %tmp.8 = call int %Ack( int %tmp.10, int 1 )            ; <int> [#uses=1]
        ret int %tmp.8

endif.1:
        %tmp.13 = add int %M.1, -1              ; <int> [#uses=1]
        %tmp.17 = add int %N.1, -1              ; <int> [#uses=1]
        %tmp.14 = call int %Ack( int %M.1, int %tmp.17 )                ; <int> [#uses=1]
        %tmp.11 = call int %Ack( int %tmp.13, int %tmp.14 )             ; <int> [#uses=1]
        ret int %tmp.11
}

