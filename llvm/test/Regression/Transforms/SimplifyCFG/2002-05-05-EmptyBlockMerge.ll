; Basic block #2 should be merged into BB #3!
;
; RUN: if as < %s | opt -dce | dis | grep 'br label'
; RUN: then exit 1
; RUN: else exit 0
; RUN: fi
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

