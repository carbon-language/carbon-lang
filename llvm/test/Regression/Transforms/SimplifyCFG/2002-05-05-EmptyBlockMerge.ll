; Basic block #2 should not be merged into BB #3!
;
; RUN: llvm-as < %s | opt -simplifycfg | llvm-dis | grep 'br label'
;
void "cprop_test12"(int* %data) {
bb0:
        %reg108 = load int* %data
        %cond218 = setne int %reg108, 5
        br bool %cond218, label %bb3, label %bb2

bb2:
        br label %bb3

bb3:
        %reg117 = phi int [ 110, %bb2 ], [ %reg108, %bb0 ]
        store int %reg117, int* %data
        ret void
}

