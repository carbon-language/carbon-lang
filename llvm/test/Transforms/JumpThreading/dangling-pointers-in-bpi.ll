; RUN: opt -S -jump-threading %s 2>&1 | FileCheck %s

; Test that after removing basic block we had also removed it from BPI. If we
; didn't there will be dangling pointer inside BPI. It can lead to a
; miscalculation in the edge weight and possibly other problems.
; Checks that we are not failing assertion inside AssettingVH

; CHECK-NOT: An asserting value handle still pointed to this value

define void @foo(i32 %n, i1 %cond) !prof !0 {
; Record this block into BPI cache
single-pred:
  br i1 %cond, label %entry, label %entry, !prof !{!"branch_weights", i32 1, i32 1}

entry:
  ret void
}

!0 = !{!"function_entry_count", i64 1}
