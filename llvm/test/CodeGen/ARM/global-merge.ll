; RUN: llc < %s -march=thumb | FileCheck %s
; Test the ARMGlobalMerge pass.  Use -march=thumb because it has a small
; value for the maximum offset (127).

; A local array that exceeds the maximum offset should not be merged.
; CHECK: g0:
@g0 = internal global [32 x i32] [ i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 1, i32 2 ]

; CHECK: merged:
@g1 = internal global i32 1
@g2 = internal global i32 2
