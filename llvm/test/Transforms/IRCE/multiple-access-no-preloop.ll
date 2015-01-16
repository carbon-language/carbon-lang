; RUN: opt -irce -S < %s | FileCheck %s

define void @multiple_access_no_preloop(
    i32* %arr_a, i32* %a_len_ptr, i32* %arr_b, i32* %b_len_ptr, i32 %n) {

 entry:
  %len.a = load i32* %a_len_ptr, !range !0
  %len.b = load i32* %b_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds.b ]
  %idx.next = add i32 %idx, 1
  %abc.a = icmp slt i32 %idx, %len.a
  br i1 %abc.a, label %in.bounds.a, label %out.of.bounds

 in.bounds.a:
  %addr.a = getelementptr i32* %arr_a, i32 %idx
  store i32 0, i32* %addr.a
  %abc.b = icmp slt i32 %idx, %len.b
  br i1 %abc.b, label %in.bounds.b, label %out.of.bounds

 in.bounds.b:
  %addr.b = getelementptr i32* %arr_b, i32 %idx
  store i32 -1, i32* %addr.b
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

; CHECK-LABEL: multiple_access_no_preloop

; CHECK-LABEL: loop.preheader:
; CHECK: [[smaller_len_cmp:[^ ]+]] = icmp slt i32 %len.a, %len.b
; CHECK: [[smaller_len:[^ ]+]] = select i1 [[smaller_len_cmp]], i32 %len.a, i32 %len.b
; CHECK: [[upper_bound_cmp:[^ ]+]] = icmp slt i32 %n, %3
; CHECK: [[upper_bound:[^ ]+]] = select i1 %5, i32 %n, i32 %3

; CHECK-LABEL: loop:
; CHECK: br i1 true, label %in.bounds.a, label %out.of.bounds

; CHECK-LABEL: in.bounds.a:
; CHECK: br i1 true, label %in.bounds.b, label %out.of.bounds

; CHECK-LABEL: in.bounds.b:
; CHECK: [[main_loop_cond:[^ ]+]] = icmp slt i32 %idx.next, [[upper_bound]]
; CHECK: br i1 [[main_loop_cond]], label %loop, label %main.exit.selector

; CHECK-LABEL: in.bounds.b.postloop:
; CHECK: %next.postloop = icmp slt i32 %idx.next.postloop, %n
; CHECK: br i1 %next.postloop, label %loop.postloop, label %exit.loopexit

!0 = !{i32 0, i32 2147483647}
