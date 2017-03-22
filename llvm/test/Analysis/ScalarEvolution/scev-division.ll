; RUN: opt < %s -analyze -scalar-evolution | FileCheck %s

declare void @llvm.experimental.guard(i1, ...)

define void @test01(i32 %a, i32 %n) nounwind {
; Prove that (n > 1) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test01
; CHECK:         Loop %header: backedge-taken count is (-1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test01neg(i32 %a, i32 %n) nounwind {
; Prove that (n > 0) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test01neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax %n.div.2))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 0
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test02(i32 %a, i32 %n) nounwind {
; Prove that (n >= 2) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test02
; CHECK:         Loop %header: backedge-taken count is (-1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test02neg(i32 %a, i32 %n) nounwind {
; Prove that (n >= 1) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @test02neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax %n.div.2))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sgt i32 %n.div.2, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test03(i32 %a, i32 %n) nounwind {
; Prove that (n > -2) ===> (n / 2 >= 0).
; TODO: We should be able to prove that (n > -2) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test03
; CHECK:         Loop %header: backedge-taken count is (1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test03neg(i32 %a, i32 %n) nounwind {
; Prove that (n > -3) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test03neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + %n.div.2)<nsw>)
entry:
  %cmp1 = icmp sgt i32 %n, -3
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test04(i32 %a, i32 %n) nounwind {
; Prove that (n >= -1) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test04
; CHECK:         Loop %header: backedge-taken count is (1 + %n.div.2)<nsw>
entry:
  %cmp1 = icmp sge i32 %n, -1
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @test04neg(i32 %a, i32 %n) nounwind {
; Prove that (n >= -2) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @test04neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + %n.div.2)<nsw>)
entry:
  %cmp1 = icmp sge i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i32 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i32 %indvar, 1
  %exitcond = icmp sge i32 %n.div.2, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext01(i32 %a, i32 %n) nounwind {
; Prove that (n > 1) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @testext01
; CHECK:         Loop %header: backedge-taken count is (-1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext01neg(i32 %a, i32 %n) nounwind {
; Prove that (n > 0) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @testext01neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax (sext i32 %n.div.2 to i64)))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, 0
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext02(i32 %a, i32 %n) nounwind {
; Prove that (n >= 2) ===> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @testext02
; CHECK:         Loop %header: backedge-taken count is (-1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext02neg(i32 %a, i32 %n) nounwind {
; Prove that (n >= 1) =\=> (n / 2 > 0).
; CHECK:         Determining loop execution counts for: @testext02neg
; CHECK:         Loop %header: backedge-taken count is (-1 + (1 smax (sext i32 %n.div.2 to i64)))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, 1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sgt i64 %n.div.2.ext, %indvar.next
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext03(i32 %a, i32 %n) nounwind {
; Prove that (n > -2) ===> (n / 2 >= 0).
; TODO: We should be able to prove that (n > -2) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @testext03
; CHECK:         Loop %header: backedge-taken count is (1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sgt i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext03neg(i32 %a, i32 %n) nounwind {
; Prove that (n > -3) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @testext03neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + (sext i32 %n.div.2 to i64))<nsw>)
entry:
  %cmp1 = icmp sgt i32 %n, -3
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext04(i32 %a, i32 %n) nounwind {
; Prove that (n >= -1) ===> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @testext04
; CHECK:         Loop %header: backedge-taken count is (1 + (sext i32 %n.div.2 to i64))<nsw>
entry:
  %cmp1 = icmp sge i32 %n, -1
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

define void @testext04neg(i32 %a, i32 %n) nounwind {
; Prove that (n >= -2) =\=> (n / 2 >= 0).
; CHECK:         Determining loop execution counts for: @testext04neg
; CHECK:         Loop %header: backedge-taken count is (0 smax (1 + (sext i32 %n.div.2 to i64))<nsw>)
entry:
  %cmp1 = icmp sge i32 %n, -2
  %n.div.2 = sdiv i32 %n, 2
  %n.div.2.ext = sext i32 %n.div.2 to i64
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %header

header:
  %indvar = phi i64 [ %indvar.next, %header ], [ 0, %entry ]
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp sge i64 %n.div.2.ext, %indvar
  br i1 %exitcond, label %header, label %exit

exit:
  ret void
}

