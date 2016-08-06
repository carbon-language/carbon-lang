target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

; Function Attrs: nounwind ssp uwtable
define linkonce_odr i32 @baz() #0 {
entry:
  ret i32 0
}

define i8* @bar() {
entry:
  ret i8* bitcast (i32 ()* @baz to i8*)
}
