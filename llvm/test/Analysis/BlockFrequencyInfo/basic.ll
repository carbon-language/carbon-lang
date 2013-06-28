; RUN: opt < %s -analyze -block-freq | FileCheck %s

define i32 @test1(i32 %i, i32* %a) {
; CHECK: Printing analysis {{.*}} for function 'test1'
; CHECK: entry = 1.0
entry:
  br label %body

; Loop backedges are weighted and thus their bodies have a greater frequency.
; CHECK: body = 32.0
body:
  %iv = phi i32 [ 0, %entry ], [ %next, %body ]
  %base = phi i32 [ 0, %entry ], [ %sum, %body ]
  %arrayidx = getelementptr inbounds i32* %a, i32 %iv
  %0 = load i32* %arrayidx
  %sum = add nsw i32 %0, %base
  %next = add i32 %iv, 1
  %exitcond = icmp eq i32 %next, %i
  br i1 %exitcond, label %exit, label %body

; CHECK: exit = 1.0
exit:
  ret i32 %sum
}

define i32 @test2(i32 %i, i32 %a, i32 %b) {
; CHECK: Printing analysis {{.*}} for function 'test2'
; CHECK: entry = 1.0
entry:
  %cond = icmp ult i32 %i, 42
  br i1 %cond, label %then, label %else, !prof !0

; The 'then' branch is predicted more likely via branch weight metadata.
; CHECK: then = 0.94116
then:
  br label %exit

; CHECK: else = 0.05877
else:
  br label %exit

; FIXME: It may be a bug that we don't sum back to 1.0.
; CHECK: exit = 0.99993
exit:
  %result = phi i32 [ %a, %then ], [ %b, %else ]
  ret i32 %result
}

!0 = metadata !{metadata !"branch_weights", i32 64, i32 4}

define i32 @test3(i32 %i, i32 %a, i32 %b, i32 %c, i32 %d, i32 %e) {
; CHECK: Printing analysis {{.*}} for function 'test3'
; CHECK: entry = 1.0
entry:
  switch i32 %i, label %case_a [ i32 1, label %case_b
                                 i32 2, label %case_c
                                 i32 3, label %case_d
                                 i32 4, label %case_e ], !prof !1

; CHECK: case_a = 0.04998
case_a:
  br label %exit

; CHECK: case_b = 0.04998
case_b:
  br label %exit

; The 'case_c' branch is predicted more likely via branch weight metadata.
; CHECK: case_c = 0.79998
case_c:
  br label %exit

; CHECK: case_d = 0.04998
case_d:
  br label %exit

; CHECK: case_e = 0.04998
case_e:
  br label %exit

; FIXME: It may be a bug that we don't sum back to 1.0.
; CHECK: exit = 0.99993
exit:
  %result = phi i32 [ %a, %case_a ],
                    [ %b, %case_b ],
                    [ %c, %case_c ],
                    [ %d, %case_d ],
                    [ %e, %case_e ]
  ret i32 %result
}

!1 = metadata !{metadata !"branch_weights", i32 4, i32 4, i32 64, i32 4, i32 4}

; CHECK: Printing analysis {{.*}} for function 'nested_loops'
; CHECK: entry = 1.0
; This test doesn't seem to be assigning sensible frequencies to nested loops.
define void @nested_loops(i32 %a) {
entry:
  br label %for.cond1.preheader

for.cond1.preheader:
  %x.024 = phi i32 [ 0, %entry ], [ %inc12, %for.inc11 ]
  br label %for.cond4.preheader

for.cond4.preheader:
  %y.023 = phi i32 [ 0, %for.cond1.preheader ], [ %inc9, %for.inc8 ]
  %add = add i32 %y.023, %x.024
  br label %for.body6

for.body6:
  %z.022 = phi i32 [ 0, %for.cond4.preheader ], [ %inc, %for.body6 ]
  %add7 = add i32 %add, %z.022
  tail call void @g(i32 %add7) #2
  %inc = add i32 %z.022, 1
  %cmp5 = icmp ugt i32 %inc, %a
  br i1 %cmp5, label %for.inc8, label %for.body6, !prof !2

for.inc8:
  %inc9 = add i32 %y.023, 1
  %cmp2 = icmp ugt i32 %inc9, %a
  br i1 %cmp2, label %for.inc11, label %for.cond4.preheader, !prof !2

for.inc11:
  %inc12 = add i32 %x.024, 1
  %cmp = icmp ugt i32 %inc12, %a
  br i1 %cmp, label %for.end13, label %for.cond1.preheader, !prof !2

for.end13:
  ret void
}

declare void @g(i32) #1

!2 = metadata !{metadata !"branch_weights", i32 1, i32 4000}
