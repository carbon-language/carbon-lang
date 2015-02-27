; RUN: opt -irce-print-changed-loops -irce -S < %s 2>&1 | FileCheck %s

; Demonstrates that we don't currently handle the general expression
; `A * I + B'.

define void @general_affine_expressions(i32 *%arr, i32 *%a_len_ptr, i32 %n,
                                        i32 %scale, i32 %offset) {
; CHECK-NOT: constrained Loop at depth
 entry:
  %len = load i32* %a_len_ptr, !range !0
  %first.itr.check = icmp sgt i32 %n, 0
  br i1 %first.itr.check, label %loop, label %exit

 loop:
  %idx = phi i32 [ 0, %entry ] , [ %idx.next, %in.bounds ]
  %idx.next = add i32 %idx, 1
  %idx.mul = mul i32 %idx, %scale
  %array.idx = add i32 %idx.mul, %offset
  %abc.high = icmp slt i32 %array.idx, %len
  %abc.low = icmp sge i32 %array.idx, 0
  %abc = and i1 %abc.low, %abc.high
  br i1 %abc, label %in.bounds, label %out.of.bounds

 in.bounds:
  %addr = getelementptr i32, i32* %arr, i32 %array.idx
  store i32 0, i32* %addr
  %next = icmp slt i32 %idx.next, %n
  br i1 %next, label %loop, label %exit

 out.of.bounds:
  ret void

 exit:
  ret void
}

!0 = !{i32 0, i32 2147483647}
