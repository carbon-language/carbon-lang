; RUN: llvm-as < %s | llc -march=c | \
; RUN:   grep 'return ((((ltmp_2_2 == (signed int)ltmp_1_2)) ?  (1) : (0)))'
; For PR1099
; XFAIL: *

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8"
        %struct.Connector = type { i16, i16, i8, i8, %struct.Connector*, i8* }

implementation   ; Functions:

define bool @prune_match_entry_2E_ce(%struct.Connector* %a, i16 %b.0.0.val) {
newFuncRoot:
        br label %entry.ce

cond_next.exitStub:             ; preds = %entry.ce
        ret bool true

entry.return_crit_edge.exitStub:                ; preds = %entry.ce
        ret bool false

entry.ce:               ; preds = %newFuncRoot
        %tmp = getelementptr %struct.Connector* %a, i32 0, i32 0                ; <i16*> [#uses=1]
        %tmp = load i16* %tmp           ; <i16> [#uses=1]
        %tmp = icmp eq i16 %tmp, %b.0.0.val             ; <bool> [#uses=1]
        br bool %tmp, label %cond_next.exitStub, label %entry.return_crit_edge.exitStub
}


