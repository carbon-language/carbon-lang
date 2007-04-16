; RUN: llvm-upgrade < %s | llvm-as | opt -tailduplicate | \
; RUN:   llc -march=x86 -o %t -f
; RUN: grep {je } %t
; RUN: not grep jmp %t
; END.
; This should have no unconditional jumps in it.  The C source is:

;void foo(int c, int* P) {
;  if (c & 1)  P[0] = 1;
;  if (c & 2)  P[1] = 1;
;  if (c & 4)  P[2] = 1;
;  if (c & 8)  P[3] = 1;
;}

implementation

void %foo(int %c, int* %P) {
entry:
        %tmp1 = and int %c, 1           ; <int> [#uses=1]
        %tmp1 = seteq int %tmp1, 0              ; <bool> [#uses=1]
        br bool %tmp1, label %cond_next, label %cond_true

cond_true:              ; preds = %entry
        store int 1, int* %P
        br label %cond_next

cond_next:              ; preds = %entry, %cond_true
        %tmp5 = and int %c, 2           ; <int> [#uses=1]
        %tmp5 = seteq int %tmp5, 0              ; <bool> [#uses=1]
        br bool %tmp5, label %cond_next10, label %cond_true6

cond_true6:             ; preds = %cond_next
        %tmp8 = getelementptr int* %P, int 1            ; <int*> [#uses=1]
        store int 1, int* %tmp8
        br label %cond_next10

cond_next10:            ; preds = %cond_next, %cond_true6
        %tmp13 = and int %c, 4          ; <int> [#uses=1]
        %tmp13 = seteq int %tmp13, 0            ; <bool> [#uses=1]
        br bool %tmp13, label %cond_next18, label %cond_true14

cond_true14:            ; preds = %cond_next10
        %tmp16 = getelementptr int* %P, int 2           ; <int*> [#uses=1]
        store int 1, int* %tmp16
        br label %cond_next18

cond_next18:            ; preds = %cond_next10, %cond_true14
        %tmp21 = and int %c, 8          ; <int> [#uses=1]
        %tmp21 = seteq int %tmp21, 0            ; <bool> [#uses=1]
        br bool %tmp21, label %return, label %cond_true22

cond_true22:            ; preds = %cond_next18
        %tmp24 = getelementptr int* %P, int 3           ; <int*> [#uses=1]
        store int 1, int* %tmp24
        ret void

return:         ; preds = %cond_next18
        ret void
}


