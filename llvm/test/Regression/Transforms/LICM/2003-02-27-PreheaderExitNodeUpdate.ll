; This testcase fails because preheader insertion is not updating exit node 
; information for loops.

; RUN: as < %s | opt -licm

int %main(int %argc, sbyte** %argv) {
bb0:            ; No predecessors!
        br bool false, label %bb7, label %bb5

bb5:            ; preds = %bb5, %bb0
        br bool false, label %bb5, label %bb7

bb7:            ; preds = %bb7, %bb5, %bb0
        br bool false, label %bb7, label %bb10

bb10:           ; preds = %bb7
        ret int 0
}

