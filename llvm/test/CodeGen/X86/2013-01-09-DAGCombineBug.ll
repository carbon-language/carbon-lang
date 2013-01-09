; RUN: llc -mtriple=x86_64-apple-macosx10.5.0 < %s

; rdar://12968664

define void @t() nounwind uwtable ssp {
  br label %4

; <label>:1                                       ; preds = %4, %2
  ret void

; <label>:2                                       ; preds = %6, %5, %3, %2
  switch i32 undef, label %2 [
    i32 1090573978, label %1
    i32 1090573938, label %3
    i32 1090573957, label %5
  ]

; <label>:3                                       ; preds = %4, %2
  br i1 undef, label %2, label %4

; <label>:4                                       ; preds = %6, %5, %3, %0
  switch i32 undef, label %11 [
    i32 1090573938, label %3
    i32 1090573957, label %5
    i32 1090573978, label %1
    i32 165205179, label %6
  ]

; <label>:5                                       ; preds = %4, %2
  br i1 undef, label %2, label %4

; <label>:6                                       ; preds = %4
  %7 = icmp eq i32 undef, 590901838
  %8 = or i1 false, %7
  %9 = or i1 true, %8
  %10 = xor i1 %8, %9
  br i1 %10, label %4, label %2

; <label>:11                                      ; preds = %11, %4
  br label %11
}
