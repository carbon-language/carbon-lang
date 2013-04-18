; RUN: opt < %s -instcombine -S | FileCheck %s

define void @f(i64 %val, i32  %limit, i32 *%ptr) {
;CHECK: %0 = trunc i64
;CHECK: %1 = phi i32
entry:
  %tempvector = insertelement <16 x i64> undef, i64 %val, i32 0
  %vector = shufflevector <16 x i64> %tempvector, <16 x i64> undef, <16 x i32> zeroinitializer
  %0 = add <16 x i64> %vector, <i64 0, i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8, i64 9, i64 10, i64 11, i64 12, i64 13, i64 14, i64 15>
  %1 = trunc <16 x i64> %0 to <16 x i32>
  br label %loop

loop:
  %2 = phi <16 x i32> [ %1, %entry ], [ %inc, %loop ]
  %elt = extractelement <16 x i32> %2, i32 0
  %end = icmp ult i32 %elt, %limit
  %3 = add i32 10, %elt
  %4 = sext i32 %elt to i64
  %5 = getelementptr i32* %ptr, i64 %4
  store i32 %3, i32* %5
  %inc = add <16 x i32> %2, <i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16, i32 16>
  br i1 %end, label %loop, label %ret

ret:
  ret void
}

