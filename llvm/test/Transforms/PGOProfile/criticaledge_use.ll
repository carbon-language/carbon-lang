; RUN: llvm-profdata merge %S/Inputs/criticaledge.proftext -o %T/criticaledge.profdata
; RUN: opt < %s -pgo-instr-use -pgo-profile-file=%T/criticaledge.profdata -S | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @_Z17test_criticalEdgeii(i32 %i, i32 %j) {
entry:
  switch i32 %i, label %sw.default [
    i32 1, label %sw.bb
    i32 2, label %sw.bb1
    i32 3, label %sw.bb3
    i32 4, label %sw.bb3
; CHECK:    i32 3, label %entry.sw.bb3_crit_edge
; CHECK:    i32 4, label %entry.sw.bb3_crit_edge1
    i32 5, label %sw.bb3
  ]
; CHECK: !prof !0

; CHECK: entry.sw.bb3_crit_edge1:
; CHECK:   br label %sw.bb3
; CHECK: entry.sw.bb3_crit_edge:
; CHECK:   br label %sw.bb3

sw.bb:
  %call = call i32 @_ZL3bari(i32 2)
  br label %sw.epilog

sw.bb1:
  %call2 = call i32 @_ZL3bari(i32 1024)
  br label %sw.epilog

sw.bb3:
  %cmp = icmp eq i32 %j, 2
  br i1 %cmp, label %if.then, label %if.end
; CHECK: !prof !1

if.then:
  %call4 = call i32 @_ZL3bari(i32 4)
  br label %return

if.end:
  %call5 = call i32 @_ZL3bari(i32 8)
  br label %sw.epilog

sw.default:
  %call6 = call i32 @_ZL3bari(i32 32)
  %cmp7 = icmp sgt i32 %j, 10
  br i1 %cmp7, label %if.then8, label %if.end9
; CHECK: !prof !2

if.then8:
  %add = add nsw i32 %call6, 10
  br label %if.end9

if.end9:
  %res.0 = phi i32 [ %add, %if.then8 ], [ %call6, %sw.default ]
  br label %sw.epilog

sw.epilog:
  %res.1 = phi i32 [ %res.0, %if.end9 ], [ %call5, %if.end ], [ %call2, %sw.bb1 ], [ %call, %sw.bb ]
  br label %return

return:
  %retval = phi i32 [ %res.1, %sw.epilog ], [ %call4, %if.then ]
  ret i32 %retval
}

define internal i32 @_ZL3bari(i32 %i) {
entry:
  ret i32 %i
}

; CHECK: !0 = !{!"branch_weights", i32 2, i32 1, i32 0, i32 2, i32 1, i32 1}
; CHECK: !1 = !{!"branch_weights", i32 2, i32 2}
; CHECK: !2 = !{!"branch_weights", i32 1, i32 1}
