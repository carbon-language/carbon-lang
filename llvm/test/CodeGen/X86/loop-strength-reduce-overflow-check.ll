; RUN: llc < %s -O1 -mtriple=x86_64-unknown-unknown

; LSR has to check for overflow to avoid UB when it examines reuse opportunities
; Clang built with UBSan would expose the issue in this test case

define void @main() {
bb:
  br label %bb1

bb1:                                              ; preds = %bb1, %bb
  %tmp = phi i64 [ 248268322795906120, %bb ], [ %tmp10, %bb1 ]
  %tmp2 = sub i64 %tmp, 248268322795906120
  %tmp3 = getelementptr i8, i8* undef, i64 %tmp2
  %tmp4 = sub i64 %tmp, 248268322795906120
  %tmp5 = getelementptr i8, i8* undef, i64 %tmp4
  %tmp6 = getelementptr i8, i8* %tmp5, i64 -9086989864993762928
  %tmp7 = load i8, i8* %tmp6, align 1
  %tmp8 = getelementptr i8, i8* %tmp3, i64 -1931736422337600660
  store i8 undef, i8* %tmp8
  %tmp9 = add i64 %tmp, 248268322795906121
  %tmp10 = add i64 %tmp9, -248268322795906120
  br i1 undef, label %bb11, label %bb1

bb11:                                             ; preds = %bb1
  ret void
}
