; RUN: opt < %s -analyze -enable-new-pm=0 -scalar-evolution | FileCheck %s
; RUN: opt < %s -disable-output "-passes=print<scalar-evolution>" 2>&1 | FileCheck %s
; PR4569

define i16 @main() nounwind {
entry:
        br label %bb.i

bb.i:           ; preds = %bb1.i, %bb.nph
; We should be able to find the range for this expression.
; CHECK: %l_95.0.i1 = phi i8
; CHECK: -->  {0,+,-1}<%bb.i> U: [2,1) S: [2,1){{ *}}Exits: 2

        %l_95.0.i1 = phi i8 [ %tmp1, %bb.i ], [ 0, %entry ]

; This cast shouldn't be folded into the addrec.
; CHECK: %tmp = zext i8 %l_95.0.i1 to i16
; CHECK: -->  (zext i8 {0,+,-1}<%bb.i> to i16){{ U: [^ ]+ S: [^ ]+}}{{ *}}Exits: 2

        %tmp = zext i8 %l_95.0.i1 to i16

        %tmp1 = add i8 %l_95.0.i1, -1
        %phitmp = icmp eq i8 %tmp1, 1
        br i1 %phitmp, label %bb1.i.func_36.exit_crit_edge, label %bb.i

bb1.i.func_36.exit_crit_edge:
        ret i16 %tmp
}
