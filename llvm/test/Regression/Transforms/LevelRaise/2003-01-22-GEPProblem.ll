; Testcase reduced from 197.parser by bugpoint
; RUN: llvm-as < %s | opt -raise -raise-start-inst=cast455 > /dev/null

void %conjunction_prune() {
; <label>:0             ; No predecessors!
        br label %bb19

bb19:           ; preds = %bb22, %0
        %reg205 = phi ulong [ %cast208, %bb22 ], [ 0, %0 ]              ; <ulong> [#uses=2]
        %reg449 = add ulong %reg205, 10         ; <ulong> [#uses=0]
        %cast455 = cast ulong %reg205 to sbyte**                ; <sbyte**> [#uses=1]
        store sbyte* null, sbyte** %cast455
        br label %bb22

bb22:           ; preds = %bb19
        %cast208 = cast sbyte* null to ulong            ; <ulong> [#uses=1]
        br bool false, label %bb19, label %bb28

bb28:           ; preds = %bb22
        ret void
}

