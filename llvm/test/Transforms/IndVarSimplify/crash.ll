; RUN: opt -indvars %s -disable-output

declare i32 @putchar(i8) nounwind

define void @t2(i1* %P) nounwind {
; <label>:0
  br label %1

; <label>:1                                       ; preds = %1, %0
  %2 = phi double [ 9.000000e+00, %0 ], [ %4, %1 ] ; <double> [#uses=1]
  %3 = tail call i32 @putchar(i8 72)              ; <i32> [#uses=0]
  %4 = fadd double %2, -1.000000e+00              ; <double> [#uses=2]
  %5 = fcmp ult double %4, 0.000000e+00           ; <i1> [#uses=1]
  store i1 %5, i1* %P
  br i1 %5, label %6, label %1

; <label>:6                                       ; preds = %1
  ret void
}
