; This test checks to see if the CEE pass is applying jump-bypassing for values
; determined by PHI nodes.  Because we are coming from a particular block, we
; know what value a PHI node will take on this edge and should exploit it.
;
; This testcase comes from the following C code:
; void bar(void);
; void foo(int c) {
;   int i = c ? 2 : 8;
;   while (i < 20) {
;     bar ();
;     i++;
;   }
; }
;
; RUN: as < %s | opt -cee -simplifycfg | dis | not grep bb3

implementation
declare void %bar()

void %foo(int %c) {
bb0:            ; No predecessors!
        %cond215 = seteq int %c, 0              ; <bool> [#uses=1]
        br bool %cond215, label %bb3, label %bb4

bb3:            ; preds = %bb0
        br label %bb4

bb4:            ; preds = %bb3, %bb0
        %reg110 = phi int [ 8, %bb3 ], [ 2, %bb0 ]              ; <int> [#uses=2]
        %cond217 = setgt int %reg110, 19                ; <bool> [#uses=1]
        br bool %cond217, label %bb6, label %bb5

bb5:            ; preds = %bb5, %bb4
        %cann-indvar = phi int [ 0, %bb4 ], [ %add1-indvar, %bb5 ]              ; <int> [#uses=2]
        %add1-indvar = add int %cann-indvar, 1          ; <int> [#uses=1]
        %reg111 = add int %cann-indvar, %reg110         ; <int> [#uses=1]
        call void %bar( )
        %reg112 = add int %reg111, 1            ; <int> [#uses=1]
        %cond222 = setle int %reg112, 19                ; <bool> [#uses=1]
        br bool %cond222, label %bb5, label %bb6

bb6:            ; preds = %bb5, %bb4
        ret void
}

