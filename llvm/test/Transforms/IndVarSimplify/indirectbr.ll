; RUN: opt < %s -indvars -S -disable-output
; PR5758

define zeroext i1 @foo() nounwind {
entry:
  indirectbr i8* undef, [label %"202", label %"133"]

"132":                                            ; preds = %"133"
  %0 = add i32 %1, 1                              ; <i32> [#uses=1]
  br label %"133"

"133":                                            ; preds = %"132", %entry
  %1 = phi i32 [ %0, %"132" ], [ 0, %entry ]      ; <i32> [#uses=2]
  %2 = icmp eq i32 %1, 4                          ; <i1> [#uses=1]
  br i1 %2, label %"134", label %"132"

"134":                                            ; preds = %"133"
  ret i1 true

"202":                                            ; preds = %entry
  ret i1 false
}
