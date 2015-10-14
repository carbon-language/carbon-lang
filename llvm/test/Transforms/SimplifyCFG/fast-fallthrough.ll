; RUN: opt -S %s -simplifycfg | FileCheck %s

define void @test(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test
  %iplus1 = add nsw i32 %i, 1
  %var29 = icmp slt i32 %i, %length.i
  %var30 = icmp slt i32 %iplus1, %length.i
; CHECK: br i1 %var30, label %in_bounds, label %next
  br i1 %var29, label %next, label %out_of_bounds, !prof !{!"branch_weights", i32 1000, i32 0}

next:
; CHECK-LABEL: next:
; CHECK: br i1 %var29, label %out_of_bounds2, label %out_of_bounds
  br i1 %var30, label %in_bounds, label %out_of_bounds2, !prof !{!"branch_weights", i32 1000, i32 0}

in_bounds:
  ret void

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 1)
  unreachable
}

define void @test2(i32 %length.i, i32 %i) {
; CHECK-LABEL: @test2
  %var29 = icmp slt i32 %i, %length.i
; CHECK: br i1 %var30, label %in_bounds, label %next
  br i1 %var29, label %next, label %out_of_bounds, !prof !{!"branch_weights", i32 1000, i32 0}

next:
; CHECK-LABEL: next:
; CHECK: br i1 %var29, label %out_of_bounds2, label %out_of_bounds
  %iplus1 = add nsw i32 %i, 1
  %var30 = icmp slt i32 %iplus1, %length.i
  br i1 %var30, label %in_bounds, label %out_of_bounds2, !prof !{!"branch_weights", i32 1000, i32 0}

in_bounds:
  ret void

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 1)
  unreachable
}

; As written, this one can't trigger today.  It would require us to duplicate
; the %val1 load down two paths and that's not implemented yet.
define i64 @test3(i32 %length.i, i32 %i, i64* %base) {
; CHECK-LABEL: @test3
  %var29 = icmp slt i32 %i, %length.i
; CHECK: br i1 %var29, label %next, label %out_of_bounds
  br i1 %var29, label %next, label %out_of_bounds, !prof !{!"branch_weights", i32 1000, i32 0}

next:
; CHECK-LABEL: next:
  %addr1 = getelementptr i64, i64* %base, i32 %i
  %val1 = load i64, i64* %addr1
  %iplus1 = add nsw i32 %i, 1
  %var30 = icmp slt i32 %iplus1, %length.i
; CHECK: br i1 %var30, label %in_bounds, label %out_of_bounds2
  br i1 %var30, label %in_bounds, label %out_of_bounds2, !prof !{!"branch_weights", i32 1000, i32 0}

in_bounds:
  %addr2 = getelementptr i64, i64* %base, i32 %iplus1
  %val2 = load i64, i64* %addr2
  %res = sub i64 %val1, %val2
  ret i64 %res

out_of_bounds:
  call void @foo(i64 0)
  unreachable

out_of_bounds2:
  call void @foo(i64 %val1)
  unreachable
}

declare void @foo(i64)

